// CPU (uw_ct_vio::KltStereoTracker) vs GPU (gpu_frontend::GpuKltStereoTracker) front-end benchmark on a rosbag2.
//
// Both trackers see the same frames in the same process. Prints a progress line every --report-every frames and a
// final table (per-stage latency and agreement). With --track-out the GPU features are written as a UWTRK1 track
// file that run_bag can replay (--track-file F --track-mode replace).
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_frontend/GpuKltStereoTracker.hpp"

using namespace uw_ct_vio;
using gpu_frontend::GpuKltStereoTracker;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kStereoPairTolS = 0.005;

struct Args {
  std::string bag, intrinsics, imucam, left = "cam1", right = "cam0", left_topic, right_topic, track_out;
  int lcn = 2, lcn_stereo = 0, max_features = 150, max_frames = 0, report_every = 100, warmup = 10;
  bool run_cpu = true, cpu_stages = true;
  bool open_loop = false;     ///< re-seed the GPU tracker with the CPU tracks after every frame
  bool keepalive = false;     ///< GPU clock keep-alive kernel (laptop DVFS)
  int debug_diff_frames = 0;  ///< print the CPU/GPU feature-set differences of the first N frames
  double resize = 1.0;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
      return argv[++i];
    };
    if (k == "--bag") a.bag = val();
    else if (k == "--intrinsics") a.intrinsics = val();
    else if (k == "--imucam") a.imucam = val();
    else if (k == "--left") a.left = val();
    else if (k == "--right") a.right = val();
    else if (k == "--left-topic") a.left_topic = val();
    else if (k == "--right-topic") a.right_topic = val();
    else if (k == "--track-out") a.track_out = val();
    else if (k == "--lcn") a.lcn = std::stoi(val());
    else if (k == "--lcn-stereo") a.lcn_stereo = std::stoi(val());
    else if (k == "--max-features") a.max_features = std::stoi(val());
    else if (k == "--max-frames") a.max_frames = std::stoi(val());
    else if (k == "--report-every") a.report_every = std::stoi(val());
    else if (k == "--warmup") a.warmup = std::stoi(val());
    else if (k == "--debug-diff") a.debug_diff_frames = std::stoi(val());
    else if (k == "--open-loop") a.open_loop = true;
    else if (k == "--gpu-keepalive") a.keepalive = true;
    else if (k == "--no-cpu") a.run_cpu = false;
    else if (k == "--no-cpu-stages") a.cpu_stages = false;
    else if (k == "--resize") a.resize = std::stod(val());
    else throw std::invalid_argument("unknown argument " + k);
  }
  for (const auto* r : {&a.bag, &a.intrinsics, &a.imucam, &a.left_topic, &a.right_topic})
    if (r->empty()) throw std::invalid_argument("need --bag --intrinsics --imucam --left-topic --right-topic");
  return a;
}

// ---- calibration (as run_bag.cpp) ----
CameraModel camera(const YAML::Node& n, const std::string& name, double s) {
  CameraModel c;
  const std::string model = n["distortion_model"].as<std::string>();
  if (n["camera_model"].as<std::string>() != "pinhole") throw std::invalid_argument(name + ": only pinhole supported");
  if (model == "equidistant") c.distortion = CameraModel::Distortion::Equidistant;
  else if (model == "radtan") c.distortion = CameraModel::Distortion::RadTan;
  else throw std::invalid_argument(name + ": unsupported distortion " + model);
  const auto in = n["intrinsics"].as<std::vector<double>>();
  const auto res = n["resolution"].as<std::vector<int>>();
  c.fx = in.at(0) * s, c.fy = in.at(1) * s, c.cx = in.at(2) * s, c.cy = in.at(3) * s;
  c.width = static_cast<int>(std::lround(res.at(0) * s)), c.height = static_cast<int>(std::lround(res.at(1) * s));
  c.coeffs = n["distortion_coeffs"].as<std::vector<double>>();
  c.validate(name);
  return c;
}

gtsam::Pose3 pose4x4(const YAML::Node& n) {
  gtsam::Matrix4 m;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) m(r, c) = n[r][c].as<double>();
  return gtsam::Pose3(m);
}

double stamp(const builtin_interfaces::msg::Time& t) { return t.sec + 1e-9 * t.nanosec; }

template <typename T>
T deserialize(const rosbag2_storage::SerializedBagMessage& m) {
  rclcpp::SerializedMessage sm(*m.serialized_data);
  T out;
  rclcpp::Serialization<T>().deserialize_message(&sm, &out);
  return out;
}

cv::Mat grey(const sensor_msgs::msg::Image& m) {
  if (m.encoding == "mono8") return cv::Mat(m.height, m.width, CV_8UC1, const_cast<uint8_t*>(m.data.data()), m.step).clone();
  if (m.encoding == "bgr8" || m.encoding == "rgb8") {
    cv::Mat c(m.height, m.width, CV_8UC3, const_cast<uint8_t*>(m.data.data()), m.step), g;
    cv::cvtColor(c, g, m.encoding == "bgr8" ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    return g;
  }
  throw std::runtime_error("unsupported image encoding " + m.encoding);
}

// ---- statistics ----
struct Series {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double mean() const { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
  double pct(double p) const {
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s[std::min(s.size() - 1, static_cast<size_t>(p * static_cast<double>(s.size() - 1) + 0.5))];
  }
  double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
};

void row(const char* name, const Series& s) {
  std::printf("  %-34s %8.2f %8.2f %8.2f %8.2f %8.2f\n", name, s.mean(), s.pct(0.5), s.pct(0.95), s.pct(0.99), s.max());
}

// ---- CPU stage replay: the same OpenCV calls as KltStereoTracker, timed on the tracker's own frames/points ----
struct CpuProbe {
  const KltConfig& cfg;
  const CameraModel& cam;
  cv::Ptr<cv::CLAHE> clahe;
  cv::Mat prevLcn;
  CpuProbe(const KltConfig& c, const CameraModel& m) : cfg(c), cam(m), clahe(cv::createCLAHE(c.clahe_clip, cv::Size(c.clahe_tiles, c.clahe_tiles))) {}

  static double noiseSigma(const cv::Mat& g8) {
    cv::Mat g, resp, gx, gy, mag;
    g8.convertTo(g, CV_32F);
    const cv::Mat k = (cv::Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
    cv::filter2D(g, resp, CV_32F, k);
    cv::Sobel(g, gx, CV_32F, 1, 0);
    cv::Sobel(g, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, mag);
    cv::Mat m = mag.reshape(1, 1).clone();
    std::nth_element(m.begin<float>(), m.begin<float>() + m.cols / 2, m.end<float>());
    const float med = m.at<float>(m.cols / 2);
    double sum = 0;
    size_t n = 0;
    for (int y = 1; y < g.rows - 1; ++y)
      for (int x = 1; x < g.cols - 1; ++x)
        if (mag.at<float>(y, x) <= med) {
          sum += std::abs(resp.at<float>(y, x));
          ++n;
        }
    return n ? std::sqrt(M_PI / 2.0) / 6.0 * sum / static_cast<double>(n) : 0.0;
  }
  cv::Mat lcn(const cv::Mat& raw, double sigma) const {
    cv::Mat f, mu, mu2, var, z, out;
    raw.convertTo(f, CV_32F);
    const cv::Size k(cfg.lcn_window_px, cfg.lcn_window_px);
    cv::boxFilter(f, mu, CV_32F, k);
    cv::boxFilter(f.mul(f), mu2, CV_32F, k);
    const double floor2 = std::pow(std::max(3.0 * std::max(sigma, 0.5), cfg.lcn_min_floor), 2);
    var = mu2 - mu.mul(mu) + floor2;
    cv::sqrt(cv::max(var, floor2), var);
    cv::divide(f - mu, var, z);
    z.convertTo(out, CV_8U, 40.0, 128.0);
    return out;
  }
  std::vector<cv::Point2f> toPixels(const std::vector<FeatureObs>& obs, const std::set<uint64_t>* only) const {
    std::vector<cv::Point3f> rays;
    for (const auto& o : obs)
      if (!only || only->count(o.track_id)) rays.emplace_back(static_cast<float>(o.left.x()), static_cast<float>(o.left.y()), 1.f);
    return cam.project(rays);
  }
  void detect(const cv::Mat& img, std::vector<cv::Point2f> pts) const {
    const int cells = cfg.grid_cols * cfg.grid_rows;
    const int perCell = std::max(1, cfg.max_features / cells);
    const double cw = static_cast<double>(img.cols) / cfg.grid_cols, ch = static_cast<double>(img.rows) / cfg.grid_rows;
    std::vector<int> count(cells, 0);
    cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
    for (const auto& p : pts) {
      const int cx = std::clamp(static_cast<int>(p.x / cw), 0, cfg.grid_cols - 1);
      const int cy = std::clamp(static_cast<int>(p.y / ch), 0, cfg.grid_rows - 1);
      ++count[cy * cfg.grid_cols + cx];
      cv::circle(mask, p, cfg.min_distance_px, cv::Scalar(0), -1);
    }
    for (int gy = 0; gy < cfg.grid_rows; ++gy)
      for (int gx = 0; gx < cfg.grid_cols; ++gx) {
        const int need = perCell - count[gy * cfg.grid_cols + gx];
        if (need <= 0) continue;
        const cv::Rect cell(static_cast<int>(gx * cw), static_cast<int>(gy * ch), static_cast<int>(cw), static_cast<int>(ch));
        std::vector<cv::Point2f> found;
        cv::goodFeaturesToTrack(img(cell), found, need, cfg.quality_level, cfg.min_distance_px, mask(cell));
      }
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    // The keep-alive kernel is resident for the whole run; lazily loading another kernel then would deadlock.
    if (a.keepalive) setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    const YAML::Node intr = YAML::LoadFile(a.intrinsics), ext = YAML::LoadFile(a.imucam);
    const CameraModel camL = camera(intr[a.left], a.left, a.resize), camR = camera(intr[a.right], a.right, a.resize);
    const gtsam::Pose3 T_BC0 = pose4x4(ext[a.left]["T_cam_imu"]).inverse(), T_BC1 = pose4x4(ext[a.right]["T_cam_imu"]).inverse();
    const gtsam::Pose3 T_C0C1 = T_BC0.between(T_BC1);
    KltConfig kc;
    kc.max_features = a.max_features;
    kc.lcn = a.lcn == 1;
    kc.lcn_auto = a.lcn == 2;
    kc.lcn_stereo = a.lcn_stereo != 0;

    std::unique_ptr<KltStereoTracker> cpu;
    if (a.run_cpu) cpu = std::make_unique<KltStereoTracker>(kc, camL, camR, T_C0C1);
    GpuKltStereoTracker gpu(kc, camL, camR, T_C0C1);
    CpuProbe probe(kc, camL);
    if (a.keepalive) gpu.core().startClockKeepAlive();
    uint64_t cpuMaxId = 0;
    std::map<uint64_t, int> firstSeenCpu, firstSeenGpu;
    Series ageCpu, ageGpu;
    std::ofstream trk;
    if (!a.track_out.empty()) {
      trk.open(a.track_out, std::ios::binary);
      if (!trk) throw std::runtime_error("cannot write " + a.track_out);
      const char hdr[16] = {'U', 'W', 'T', 'R', 'K', '1', 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
      trk.write(hdr, sizeof(hdr));
    }

    // Timing series (ms/frame, after warm-up).
    Series cpuTotal, gpuTotal, gpuPre, gpuTemporal, gpuRansac, gpuDetect, gpuStereo, gpuPost;
    Series cpuPre, cpuTemporal, cpuDetect, cpuStereo;
    // Agreement.
    Series nCpu, nGpu, sCpu, sGpu, pxDiff, pxDiffRight, framePresent;
    size_t statsEqualFrames = 0, statsFrames = 0;
    size_t cpuFeat = 0, idPresent = 0, d01 = 0, d05 = 0, posPresent05 = 0, stereoAgree = 0, stereoBoth = 0;
    Series win;  // last report window
    Series winG;
    std::vector<FeatureObs> prevCpu;

    rosbag2_cpp::Reader reader;
    reader.open(a.bag);
    std::map<double, cv::Mat> pendL, pendR;
    int frame = 0;
    auto run = [&](double tl, const cv::Mat& l, const cv::Mat& r) {
      std::vector<FeatureObs> oc, og;
      double cMs = 0;
      if (cpu) {
        const auto t0 = Clock::now();
        oc = cpu->track(l, r);
        cMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      const auto t1 = Clock::now();
      og = gpu.track(l, r);
      const double gMs = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
      const auto& tm = gpu.lastTiming();
      if (trk) {
        const uint32_t n = static_cast<uint32_t>(og.size());
        trk.write(reinterpret_cast<const char*>(&tl), sizeof(double));
        trk.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
        for (const auto& o : og) {
          const float f[4] = {static_cast<float>(o.left.x()), static_cast<float>(o.left.y()),
                              o.has_right ? static_cast<float>(o.right.x()) : 0.f, o.has_right ? static_cast<float>(o.right.y()) : 0.f};
          const uint8_t hr = o.has_right ? 1 : 0;
          trk.write(reinterpret_cast<const char*>(&o.track_id), 8);
          trk.write(reinterpret_cast<const char*>(f), 16);
          trk.write(reinterpret_cast<const char*>(&hr), 1);
        }
      }
      if (cpu && frame < a.debug_diff_frames) {
        const auto& sc = cpu->lastStats();
        const auto& sg = gpu.lastStats();
        std::printf("  f%d stats cpu: tracked %d fb_rej %d ransac_rej %d detected %d | gpu: tracked %d fb_rej %d ransac_rej %d detected %d\n", frame,
                    sc.tracked, sc.fb_rejected, sc.ransac_rejected, sc.detected, sg.tracked, sg.fb_rejected, sg.ransac_rejected, sg.detected);
        std::map<uint64_t, const FeatureObs*> byId;
        for (const auto& o : og) byId[o.track_id] = &o;
        for (const auto& o : oc) {
          const auto it = byId.find(o.track_id);
          const double px = camL.fx * o.left.x() + camL.cx, py = camL.fy * o.left.y() + camL.cy;
          if (it == byId.end()) std::printf("  f%d CPU-only id %llu at ~(%.1f, %.1f)\n", frame, (unsigned long long)o.track_id, px, py);
          else if (o.has_right != it->second->has_right || camL.fx * (o.left - it->second->left).norm() > 0.2)
            std::printf("  f%d id %llu differs: left diff %.3f px, has_right cpu %d gpu %d\n", frame, (unsigned long long)o.track_id,
                        camL.fx * (o.left - it->second->left).norm(), o.has_right, it->second->has_right);
        }
        std::set<uint64_t> cids;
        for (const auto& o : oc) cids.insert(o.track_id);
        for (const auto& o : og)
          if (!cids.count(o.track_id))
            std::printf("  f%d GPU-only id %llu at ~(%.1f, %.1f)\n", frame, (unsigned long long)o.track_id, camL.fx * o.left.x() + camL.cx, camL.fy * o.left.y() + camL.cy);
      }
      if (frame >= a.warmup) {
        gpuTotal.add(gMs);
        gpuPre.add(tm.gpu_pre_ms);
        if (tm.gpu_temporal_ms > 0) gpuTemporal.add(tm.gpu_temporal_ms);
        if (tm.host_ransac_ms > 0) gpuRansac.add(tm.host_ransac_ms);
        if (tm.gpu_detect_ms > 0) gpuDetect.add(tm.gpu_detect_ms);
        gpuStereo.add(tm.gpu_stereo_ms);
        gpuPost.add(tm.host_post_ms);
        nGpu.add(static_cast<double>(og.size()));
        sGpu.add(static_cast<double>(gpu.lastStats().stereo));
        if (cpu) {
          cpuTotal.add(cMs);
          nCpu.add(static_cast<double>(oc.size()));
          sCpu.add(static_cast<double>(cpu->lastStats().stereo));
          win.add(cMs);
          std::map<uint64_t, const FeatureObs*> byId;
          for (const auto& o : og) byId[o.track_id] = &o;
          size_t present = 0;
          for (const auto& o : oc) {
            ++cpuFeat;
            const auto it = byId.find(o.track_id);
            if (it != byId.end()) {
              ++idPresent;
              ++present;
              const double d = camL.fx * (o.left - it->second->left).norm();
              pxDiff.add(d);
              d01 += d < 0.1;
              d05 += d < 0.5;
              ++stereoBoth;
              stereoAgree += o.has_right == it->second->has_right;
              if (o.has_right && it->second->has_right) pxDiffRight.add(camR.fx * (o.right - it->second->right).norm());
            }
            double best = 1e9;
            for (const auto& g : og) best = std::min(best, camL.fx * (o.left - g.left).norm());
            posPresent05 += best < 0.5;
          }
          {
            const auto& c = cpu->lastStats();
            const auto& g = gpu.lastStats();
            ++statsFrames;
            statsEqualFrames += c.tracked == g.tracked && c.fb_rejected == g.fb_rejected && c.ransac_rejected == g.ransac_rejected &&
                                c.detected == g.detected && c.stereo == g.stereo && c.stereo_rejected == g.stereo_rejected;
          }
          framePresent.add(oc.empty() ? 1.0 : static_cast<double>(present) / static_cast<double>(oc.size()));
          if (a.cpu_stages && !prevCpu.empty()) {
            // Replay of the CPU front-end's main OpenCV calls on this frame (tracker internals are private).
            const bool lcnOn = cpu->lastStats().lcn_active;
            const auto p0 = Clock::now();
            const double sg = lcnOn ? CpuProbe::noiseSigma(l) : 0.0;
            cv::Mat tl_, cl, cr;
            if (lcnOn) tl_ = probe.lcn(l, sg);
            probe.clahe->apply(l, cl);
            probe.clahe->apply(r, cr);
            if (!lcnOn) tl_ = cl;
            const double pre = std::chrono::duration<double, std::milli>(Clock::now() - p0).count();
            std::set<uint64_t> prevIds;
            for (const auto& o : prevCpu) prevIds.insert(o.track_id);
            const auto prevPx = probe.toPixels(prevCpu, nullptr);
            double tem = 0;
            if (!probe.prevLcn.empty() && !prevPx.empty()) {
              std::vector<cv::Point2f> fwd, back;
              std::vector<uchar> s1, s2;
              std::vector<float> err;
              const cv::Size w(kc.lk_window_px, kc.lk_window_px);
              const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
              const auto t = Clock::now();
              cv::calcOpticalFlowPyrLK(probe.prevLcn, tl_, prevPx, fwd, s1, err, w, kc.lk_levels, crit);
              cv::calcOpticalFlowPyrLK(tl_, probe.prevLcn, fwd, back, s2, err, w, kc.lk_levels, crit);
              tem = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
            }
            std::set<uint64_t> tracked;
            for (const auto& o : oc)
              if (prevIds.count(o.track_id)) tracked.insert(o.track_id);
            const auto trackedPx = probe.toPixels(oc, &tracked);
            const auto td = Clock::now();
            probe.detect(tl_, trackedPx);
            const double det = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
            const auto curPx = probe.toPixels(oc, nullptr);
            double ste = 0;
            if (!curPx.empty()) {
              const cv::Mat& sl = (lcnOn && !kc.lcn_stereo) ? cl : tl_;
              const cv::Mat& sr = (lcnOn && !kc.lcn_stereo) ? cr : cr;
              std::vector<cv::Point2f> rr, back;
              std::vector<uchar> s1, s2;
              std::vector<float> err;
              const cv::Size w(kc.lk_window_px, kc.lk_window_px);
              const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
              const auto t = Clock::now();
              cv::calcOpticalFlowPyrLK(sl, sr, curPx, rr, s1, err, w, kc.stereo_lk_levels, crit);
              cv::calcOpticalFlowPyrLK(sr, sl, rr, back, s2, err, w, kc.stereo_lk_levels, crit);
              ste = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
            }
            cpuPre.add(pre);
            if (tem > 0) cpuTemporal.add(tem);
            cpuDetect.add(det);
            cpuStereo.add(ste);
          }
          probe.prevLcn = cv::Mat();
        }
        winG.add(gMs);
      }
      if (cpu && a.cpu_stages) {
        // Keep the previous frame's tracking image and features for the next replay.
        const bool lcnOn = cpu->lastStats().lcn_active;
        probe.prevLcn = lcnOn ? probe.lcn(l, CpuProbe::noiseSigma(l)) : cv::Mat();
        if (!lcnOn) probe.clahe->apply(l, probe.prevLcn);
        prevCpu = oc;
      }
      if (cpu) {
        double sc = 0, sg = 0;
        for (const auto& o : oc) {
          cpuMaxId = std::max(cpuMaxId, o.track_id);
          sc += frame - firstSeenCpu.emplace(o.track_id, frame).first->second;
        }
        for (const auto& o : og) sg += frame - firstSeenGpu.emplace(o.track_id, frame).first->second;
        if (frame >= a.warmup && !oc.empty() && !og.empty()) {
          ageCpu.add(sc / static_cast<double>(oc.size()));
          ageGpu.add(sg / static_cast<double>(og.size()));
        }
        if (a.open_loop) {
          std::vector<uint64_t> ids;
          for (const auto& o : oc) ids.push_back(o.track_id);
          gpu.setTracks(probe.toPixels(oc, nullptr), ids, cpuMaxId + 1);
          firstSeenGpu = firstSeenCpu;
        }
      }
      ++frame;
      if (frame % a.report_every == 0) {
        std::printf("frame %5d  cpu %6.2f ms (win mean)  gpu %6.2f ms  feats cpu %3zu gpu %3zu  stereo cpu %3d gpu %3d  id-agree %5.1f%%\n",
                    frame, win.mean(), winG.mean(), oc.size(), og.size(), cpu ? cpu->lastStats().stereo : 0,
                    gpu.lastStats().stereo, framePresent.v.empty() ? 0.0 : 100.0 * framePresent.v.back());
        std::fflush(stdout);
        win = Series();
        winG = Series();
      }
    };

    while (reader.has_next()) {
      const auto m = reader.read_next();
      if (m->topic_name != a.left_topic && m->topic_name != a.right_topic) continue;
      const auto im = deserialize<sensor_msgs::msg::Image>(*m);
      const double t = stamp(im.header.stamp);
      cv::Mat g = grey(im);
      if (a.resize != 1.0) cv::resize(g, g, cv::Size(), a.resize, a.resize, cv::INTER_LINEAR);
      const bool isLeft = m->topic_name == a.left_topic;
      auto& mine = isLeft ? pendL : pendR;
      auto& other = isLeft ? pendR : pendL;
      auto it = other.lower_bound(t - kStereoPairTolS);
      if (it != other.end() && std::abs(it->first - t) < kStereoPairTolS) {
        const double tl = isLeft ? t : it->first;
        const cv::Mat l = isLeft ? g : it->second, r = isLeft ? it->second : g;
        other.erase(other.begin(), std::next(it));
        run(tl, l, r);
        if (a.max_frames > 0 && frame >= a.max_frames) break;
      } else {
        mine[t] = g;
        while (!mine.empty() && mine.begin()->first < t - 1.0) mine.erase(mine.begin());
      }
    }

    std::printf("\n=== %d frames (first %d excluded from timing), lcn=%d lcn_stereo=%d, %dx%d ===\n", frame, a.warmup, a.lcn,
                a.lcn_stereo, camL.width, camL.height);
    std::printf("latency [ms/frame]                       mean      P50      P95      P99      max\n");
    if (cpu) {
      row("CPU total track()", cpuTotal);
      if (a.cpu_stages) {
        std::printf("  CPU stages (replayed OpenCV calls, same frames/points; do not sum to total):\n");
        row("  CPU preprocessing (noise,LCN,CLAHE)", cpuPre);
        row("  CPU temporal LK fwd+bwd", cpuTemporal);
        row("  CPU detect (goodFeaturesToTrack)", cpuDetect);
        row("  CPU stereo LK fwd+bwd", cpuStereo);
      }
    }
    row("GPU total track()", gpuTotal);
    row("  GPU pre (upload,LCN/CLAHE,pyr)", gpuPre);
    row("  GPU temporal LK (event time)", gpuTemporal);
    row("  host FB+RANSAC (wall)", gpuRansac);
    row("  GPU detect (event time)", gpuDetect);
    row("  GPU stereo LK (event time)", gpuStereo);
    row("  host undistort+epipolar (wall)", gpuPost);
    if (cpu) {
      std::printf("\nspeed-up total: mean %.2fx  P50 %.2fx  P95 %.2fx  P99 %.2fx\n", cpuTotal.mean() / gpuTotal.mean(),
                  cpuTotal.pct(0.5) / gpuTotal.pct(0.5), cpuTotal.pct(0.95) / gpuTotal.pct(0.95), cpuTotal.pct(0.99) / gpuTotal.pct(0.99));
      std::printf("\nagreement (closed loop: each tracker tracks its own features)\n");
      std::printf("  mode: %s\n", a.open_loop ? "OPEN LOOP (GPU re-seeded with the CPU tracks after every frame: one-frame agreement)" : "closed loop (each tracker follows its own features; ids diverge after the first numerical difference)");
      std::printf("  mean feature age [frames]  CPU %.2f  GPU %.2f\n", ageCpu.mean(), ageGpu.mean());
      std::printf("  tracked/frame     CPU %.1f  GPU %.1f   | with stereo  CPU %.1f  GPU %.1f\n", nCpu.mean(), nGpu.mean(), sCpu.mean(), sGpu.mean());
      std::printf("  CPU features with same track id in GPU output: %.2f%%  (per-frame mean %.2f%%)\n",
                  100.0 * static_cast<double>(idPresent) / static_cast<double>(cpuFeat), 100.0 * framePresent.mean());
      std::printf("  of the CPU features: id-matched with diff < 0.1 px %.2f%%, < 0.5 px %.2f%%; any GPU feature within 0.5 px %.2f%%\n",
                  100.0 * static_cast<double>(d01) / static_cast<double>(cpuFeat), 100.0 * static_cast<double>(d05) / static_cast<double>(cpuFeat),
                  100.0 * static_cast<double>(posPresent05) / static_cast<double>(cpuFeat));
      std::printf("  frames with identical tracked/fb_rejected/ransac_rejected/detected/stereo counts: %.2f%%\n", 100.0 * static_cast<double>(statsEqualFrames) / static_cast<double>(std::max<size_t>(statsFrames, 1)));
      std::printf("  id-matched left pixel diff [px]: median %.4f  P95 %.4f  P99 %.4f  max %.3f\n", pxDiff.pct(0.5), pxDiff.pct(0.95), pxDiff.pct(0.99), pxDiff.max());
      std::printf("  id-matched right pixel diff [px] (both stereo): median %.4f  P95 %.4f\n", pxDiffRight.pct(0.5), pxDiffRight.pct(0.95));
      std::printf("  has_right flag agrees on %.2f%% of id-matched features\n", 100.0 * static_cast<double>(stereoAgree) / static_cast<double>(std::max<size_t>(stereoBoth, 1)));
    }
    if (trk) std::printf("\nwrote GPU tracks to %s\n", a.track_out.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
