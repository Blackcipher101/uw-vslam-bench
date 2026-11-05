#include "gpu_frontend/GpuKltStereoTracker.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace gpu_frontend {

using uw_ct_vio::CameraModel;
using uw_ct_vio::FeatureObs;
using uw_ct_vio::KltConfig;

namespace {

constexpr double kEssentialConfidence = 0.999;  // same constants as KltStereoTracker.cpp
constexpr int kMinPointsForRansac = 8;
constexpr int kMaxPoints = 2048;  // LK capacity; exceeding it throws

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

CoreParams makeParams(const KltConfig& cfg, const CameraModel& cam) {
  CoreParams p;
  p.width = cam.width;
  p.height = cam.height;
  p.lk_window = cfg.lk_window_px;
  p.lk_levels = cfg.lk_levels;
  p.stereo_levels = cfg.stereo_lk_levels;
  p.max_points = kMaxPoints;
  p.grid_cols = cfg.grid_cols;
  p.grid_rows = cfg.grid_rows;
  p.per_cell = std::max(1, cfg.max_features / (cfg.grid_cols * cfg.grid_rows));
  p.min_distance = cfg.min_distance_px;
  p.quality_level = cfg.quality_level;
  p.clahe_clip = cfg.clahe_clip;
  p.clahe_tiles = cfg.clahe_tiles;
  p.lcn_window = cfg.lcn_window_px;
  return p;
}

double norm2(const cv::Point2f& d) { return std::sqrt(static_cast<double>(d.x) * d.x + static_cast<double>(d.y) * d.y); }

/// Zero-mean NCC of two equal-size float patches; -1 if either is flat (as KltStereoTracker.cpp).
double zncc(const cv::Mat& a, const cv::Mat& b) {
  cv::Scalar ma, sa, mb, sb;
  cv::meanStdDev(a, ma, sa);
  cv::meanStdDev(b, mb, sb);
  if (sa[0] < 1e-6 || sb[0] < 1e-6) return -1.0;
  return cv::mean((a - ma[0]).mul(b - mb[0]))[0] / (sa[0] * sb[0]);
}

}  // namespace

GpuKltStereoTracker::GpuKltStereoTracker(const KltConfig& cfg, const CameraModel& left, const CameraModel& right,
                                         const gtsam::Pose3& T_C0C1)
    : cfg_(cfg), left_(left), right_(right), core_(makeParams(cfg, left)) {
  left_.validate("left");
  right_.validate("right");
  if (cfg_.map_predict || cfg_.reacquire || cfg_.snr_sigma || cfg_.ncc_noise_aware)
    throw std::invalid_argument(
        "GpuKltStereoTracker: map_predict, reacquire, snr_sigma and ncc_noise_aware are not implemented");
  if (left.width != right.width || left.height != right.height)
    throw std::invalid_argument("GpuKltStereoTracker: left and right cameras must have the same size");
  const gtsam::Pose3 T_C1C0 = T_C0C1.inverse();
  const gtsam::Matrix3 E = gtsam::skewSymmetric(T_C1C0.translation()) * T_C1C0.rotation().matrix();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) E_(r, c) = E(r, c);
}

void GpuKltStereoTracker::setTracks(const std::vector<cv::Point2f>& px, const std::vector<uint64_t>& ids,
                                    uint64_t nextId) {
  if (px.size() != ids.size()) throw std::invalid_argument("GpuKltStereoTracker::setTracks: size mismatch");
  pts_ = px;
  ids_ = ids;
  nextId_ = nextId;
}

std::vector<FeatureObs> GpuKltStereoTracker::track(const cv::Mat& leftIn, const cv::Mat& rightIn,
                                                    const uw_ct_vio::MapPrediction*) {
  const int W = left_.width, H = left_.height;
  if (leftIn.type() != CV_8UC1 || rightIn.type() != CV_8UC1 || leftIn.size() != rightIn.size())
    throw std::invalid_argument("GpuKltStereoTracker: expects two 8-bit grey images of equal size");
  if (leftIn.cols != W || leftIn.rows != H)
    throw std::invalid_argument("GpuKltStereoTracker: image size differs from the camera model size");
  const auto tTotal = Clock::now();
  stats_ = uw_ct_vio::KltStereoTracker::Stats();
  timing_ = Timing();

  if (!lcnDecided_) {
    if (!cfg_.lcn_auto) {
      lcnActive_ = cfg_.lcn;
      lcnDecided_ = true;
    } else {
      int hist[256] = {0};
      for (int y = 0; y < leftIn.rows; ++y)
        for (int x = 0; x < leftIn.cols; ++x) ++hist[leftIn.at<uchar>(y, x)];
      const double total = static_cast<double>(leftIn.total());
      double acc = 0;
      int p1 = 0, p99 = 255;
      for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc < 0.01 * total) p1 = v + 1;
        if (acc >= 0.99 * total) { p99 = v; break; }
      }
      autoRanges_.push_back(p99 - p1);
      if (static_cast<int>(autoRanges_.size()) >= cfg_.lcn_auto_frames) {
        std::vector<double> r = autoRanges_;
        std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
        autoRange_ = r[r.size() / 2];
        lcnActive_ = autoRange_ < cfg_.lcn_auto_max_range;
        lcnDecided_ = true;
        // Re-express the previous frame in the new representation so tracks continue cleanly.
        if (lcnActive_ && havePrev_) core_.reseedCurAsLcn(prevRaw_.data, cfg_.lcn_min_floor);
      }
    }
  }
  stats_.lcn_active = lcnActive_;
  stats_.auto_range = autoRange_;

  leftIn.copyTo(cv::Mat(H, W, CV_8UC1, core_.leftBuffer()));
  rightIn.copyTo(cv::Mat(H, W, CV_8UC1, core_.rightBuffer()));
  core_.mark(Mark::FrameStart);
  core_.upload();
  core_.frame(lcnActive_, cfg_.lcn_stereo, cfg_.lcn_min_floor);
  core_.mark(Mark::PreDone);
  bool ranTemporal = false, ranDetect = false, ranStereo = false;

  // Temporal tracking with forward-backward check and essential-matrix RANSAC.
  if (havePrev_ && !pts_.empty()) {
    if (static_cast<int>(pts_.size()) > kMaxPoints) throw std::runtime_error("GpuKltStereoTracker: too many features");
    std::vector<Pt> in(pts_.size());
    for (size_t i = 0; i < pts_.size(); ++i) in[i] = Pt{pts_[i].x, pts_[i].y};
    core_.trackTemporal(in.data(), static_cast<int>(in.size()));
    core_.mark(Mark::TemporalDone);
    ranTemporal = true;
    core_.sync();
    const auto tHost = Clock::now();
    const Pt* fwd = core_.fwd();
    const Pt* back = core_.back();
    const uint8_t* s1 = core_.statusFwd();
    const uint8_t* s2 = core_.statusBack();
    std::vector<cv::Point2f> keptPrev, keptCur;
    std::vector<uint64_t> keptIds;
    for (size_t i = 0; i < pts_.size(); ++i) {
      const bool inside = fwd[i].x >= 0 && fwd[i].y >= 0 && fwd[i].x < W && fwd[i].y < H;
      const cv::Point2f f(fwd[i].x, fwd[i].y), b(back[i].x, back[i].y);
      if (!s1[i] || !s2[i] || !inside || norm2(b - pts_[i]) > cfg_.fb_max_px) {
        ++stats_.fb_rejected;
        continue;
      }
      if (cfg_.ncc_min > 0.0) {
        const cv::Size ps(cfg_.ncc_patch_px, cfg_.ncc_patch_px);
        cv::Mat a, bb;
        cv::getRectSubPix(prevRaw_, ps, pts_[i], a, CV_32F);
        cv::getRectSubPix(leftIn, ps, f, bb, CV_32F);
        if (zncc(a, bb) < cfg_.ncc_min) {
          ++stats_.ncc_rejected;
          continue;
        }
      }
      keptPrev.push_back(pts_[i]);
      keptCur.push_back(f);
      keptIds.push_back(ids_[i]);
    }
    pts_.clear();
    ids_.clear();
    if (static_cast<int>(keptCur.size()) >= kMinPointsForRansac) {
      const auto a = left_.undistort(keptPrev), b = left_.undistort(keptCur);
      cv::Mat inl;
      cv::findEssentialMat(a, b, cv::Mat::eye(3, 3, CV_64F), cv::RANSAC, kEssentialConfidence,
                           cfg_.ransac_px / left_.fx, inl);
      for (size_t i = 0; i < keptCur.size(); ++i) {
        if (!inl.empty() && !inl.at<uchar>(static_cast<int>(i))) {
          ++stats_.ransac_rejected;
          continue;
        }
        pts_.push_back(keptCur[i]);
        ids_.push_back(keptIds[i]);
      }
    } else {
      pts_ = keptCur;
      ids_ = keptIds;
    }
    stats_.tracked = static_cast<int>(pts_.size());
    timing_.host_ransac_ms = msSince(tHost);
  }

  // New features in grid cells below their share.
  if (static_cast<int>(pts_.size()) < cfg_.max_features) {
    const int cells = cfg_.grid_cols * cfg_.grid_rows;
    const int perCell = std::max(1, cfg_.max_features / cells);
    const double cw = static_cast<double>(W) / cfg_.grid_cols, ch = static_cast<double>(H) / cfg_.grid_rows;
    std::vector<int> count(cells, 0);
    cv::Mat mask(H, W, CV_8UC1, core_.maskBuffer());
    mask.setTo(cv::Scalar(255));
    for (const auto& p : pts_) {
      const int cx = std::clamp(static_cast<int>(p.x / cw), 0, cfg_.grid_cols - 1);
      const int cy = std::clamp(static_cast<int>(p.y / ch), 0, cfg_.grid_rows - 1);
      ++count[cy * cfg_.grid_cols + cx];
      cv::circle(mask, p, cfg_.min_distance_px, cv::Scalar(0), -1);
    }
    int* need = core_.needBuffer();
    for (int c = 0; c < cells; ++c) need[c] = std::max(0, perCell - count[c]);
    core_.mark(Mark::DetectStart);
    core_.detect();
    core_.mark(Mark::DetectDone);
    ranDetect = true;
    core_.sync();
    for (int c = 0; c < cells; ++c)
      for (int k = 0; k < core_.detCount()[c]; ++k) {
        const Pt q = core_.detPts()[c * perCell + k];
        pts_.emplace_back(q.x, q.y);
        ids_.push_back(nextId_++);
        ++stats_.detected;
      }
  }
  havePrev_ = true;
  prevRaw_ = leftIn.clone();

  // Stereo: LK left -> right, forward-backward and epipolar checks.
  std::vector<FeatureObs> out(pts_.size());
  if (!pts_.empty()) {
    if (static_cast<int>(pts_.size()) > kMaxPoints) throw std::runtime_error("GpuKltStereoTracker: too many features");
    std::vector<Pt> in(pts_.size());
    for (size_t i = 0; i < pts_.size(); ++i) in[i] = Pt{pts_[i].x, pts_[i].y};
    core_.mark(Mark::StereoStart);
    core_.trackStereo(in.data(), static_cast<int>(in.size()));
    core_.mark(Mark::StereoDone);
    ranStereo = true;
    // Host work that does not need the result overlaps the GPU: left undistortion.
    const auto nl = left_.undistort(pts_);
    core_.sync();
    const auto tHost = Clock::now();
    const Pt* r = core_.fwd();
    const Pt* back = core_.back();
    const uint8_t* s1 = core_.statusFwd();
    const uint8_t* s2 = core_.statusBack();
    std::vector<cv::Point2f> rp(pts_.size());
    for (size_t i = 0; i < pts_.size(); ++i) rp[i] = cv::Point2f(r[i].x, r[i].y);
    const auto nr = right_.undistort(rp);
    const double epiMax = cfg_.epipolar_px / right_.fx;
    for (size_t i = 0; i < pts_.size(); ++i) {
      out[i].track_id = ids_[i];
      out[i].left = gtsam::Point2(nl[i].x, nl[i].y);
      const cv::Point2f b(back[i].x, back[i].y);
      if (!s1[i] || !s2[i] || norm2(b - pts_[i]) > cfg_.fb_max_px) continue;
      const cv::Vec3d x0(nl[i].x, nl[i].y, 1.0), x1(nr[i].x, nr[i].y, 1.0);
      const cv::Vec3d line = E_ * x0;
      const double d = std::abs(x1.dot(line)) / std::hypot(line[0], line[1]);
      if (!(d < epiMax)) {
        ++stats_.stereo_rejected;
        continue;
      }
      out[i].has_right = true;
      out[i].right = gtsam::Point2(nr[i].x, nr[i].y);
      ++stats_.stereo;
    }
    timing_.host_post_ms = msSince(tHost);
  }
  core_.sync();
  timing_.gpu_pre_ms = core_.elapsedMs(Mark::FrameStart, Mark::PreDone);
  if (ranTemporal) timing_.gpu_temporal_ms = core_.elapsedMs(Mark::PreDone, Mark::TemporalDone);
  if (ranDetect) timing_.gpu_detect_ms = core_.elapsedMs(Mark::DetectStart, Mark::DetectDone);
  if (ranStereo) timing_.gpu_stereo_ms = core_.elapsedMs(Mark::StereoStart, Mark::StereoDone);
  if (lcnActive_) stats_.noise_sigma = core_.lastSigma();
  timing_.total_ms = msSince(tTotal);
  return out;
}

}  // namespace gpu_frontend
