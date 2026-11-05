// Unit tests of the CUDA front end against OpenCV / the CPU KltStereoTracker. Needs a CUDA device.
#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "gpu_frontend/GpuKltCore.hpp"
#include "gpu_frontend/GpuKltStereoTracker.hpp"

using namespace gpu_frontend;

namespace {

constexpr int kW = 720, kH = 540;

/// Smooth random texture with a gradient ramp, 8-bit; deterministic.
cv::Mat texture(int w, int h, uint64_t seed, double sigma = 2.0, double contrast = 30.0) {
  cv::RNG rng(seed);
  cv::Mat n(h, w, CV_32F), b;
  rng.fill(n, cv::RNG::NORMAL, 0.0, 1.0);
  cv::GaussianBlur(n, b, cv::Size(0, 0), sigma);
  cv::normalize(b, b, -1.0, 1.0, cv::NORM_MINMAX);
  cv::Mat ramp(h, w, CV_32F);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) ramp.at<float>(y, x) = 90.f + 40.f * x / w + 20.f * y / h;
  cv::Mat out;
  cv::Mat sum = ramp + contrast * b;
  sum.convertTo(out, CV_8U);
  return out;
}

CoreParams params() {
  CoreParams p;
  p.width = kW;
  p.height = kH;
  p.max_points = 512;
  return p;
}

void feed(GpuKltCore& core, const cv::Mat& l, const cv::Mat& r) {
  l.copyTo(cv::Mat(kH, kW, CV_8UC1, core.leftBuffer()));
  r.copyTo(cv::Mat(kH, kW, CV_8UC1, core.rightBuffer()));
  core.upload();
}

cv::Mat download(GpuKltCore& core, PyrId id, int level) {
  int w, h;
  core.levelSize(id, level, &w, &h);
  cv::Mat m(h, w, CV_8UC1);
  core.downloadLevel(id, level, m.data);
  return m;
}

// ---- CPU references copied from KltStereoTracker.cpp (anonymous there) ----
double cpuNoiseSigma(const cv::Mat& g8) {
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

cv::Mat cpuLcn(const cv::Mat& raw, double sigmaNoise, int win, double minFloor) {
  cv::Mat f, mu, mu2, var, z, out;
  raw.convertTo(f, CV_32F);
  const cv::Size k(win, win);
  cv::boxFilter(f, mu, CV_32F, k);
  cv::boxFilter(f.mul(f), mu2, CV_32F, k);
  const double floor2 = std::pow(std::max(3.0 * std::max(sigmaNoise, 0.5), minFloor), 2);
  var = mu2 - mu.mul(mu) + floor2;
  cv::sqrt(cv::max(var, floor2), var);
  cv::divide(f - mu, var, z);
  z.convertTo(out, CV_8U, 40.0, 128.0);
  return out;
}

}  // namespace

TEST(Pyramid, MatchesCvPyrDownExactlyAndScharr) {
  GpuKltCore core(params());
  const cv::Mat img = texture(kW, kH, 1);
  core.loadPyramid(PyrId::Cur, img.data);
  EXPECT_EQ(core.levelCount(PyrId::Cur), 5);
  cv::Mat ref = img;
  for (int l = 0; l < 5; ++l) {
    if (l > 0) cv::pyrDown(ref, ref);
    const cv::Mat g = download(core, PyrId::Cur, l);
    ASSERT_EQ(g.size(), ref.size()) << "level " << l;
    EXPECT_EQ(cv::countNonZero(g != ref), 0) << "pyrDown mismatch at level " << l;
    cv::Mat dx, dy;
    cv::Scharr(ref, dx, CV_16S, 1, 0, 1, 0, cv::BORDER_REFLECT_101);
    cv::Scharr(ref, dy, CV_16S, 0, 1, 1, 0, cv::BORDER_REFLECT_101);
    std::vector<int16_t> d(static_cast<size_t>(ref.total()) * 2);
    core.downloadDeriv(PyrId::Cur, l, d.data());
    long bad = 0;
    for (int y = 0; y < ref.rows; ++y)
      for (int x = 0; x < ref.cols; ++x) {
        const size_t i = (static_cast<size_t>(y) * ref.cols + x) * 2;
        bad += d[i] != dx.at<int16_t>(y, x) || d[i + 1] != dy.at<int16_t>(y, x);
      }
    EXPECT_EQ(bad, 0) << "Scharr mismatch at level " << l;
  }
}

TEST(Clahe, MatchesCv) {
  GpuKltCore core(params());
  const cv::Mat l = texture(kW, kH, 2, 3.0, 10.0), r = texture(kW, kH, 3, 3.0, 10.0);
  feed(core, l, r);
  core.frame(false, true, 0.0);
  cv::Ptr<cv::CLAHE> c = cv::createCLAHE(1.0, cv::Size(2, 2));
  cv::Mat refL, refR;
  c->apply(l, refL);
  c->apply(r, refR);
  cv::Mat gl = download(core, PyrId::Cur, 0), gr = download(core, PyrId::Right, 0);
  double maxd = 0;
  cv::minMaxLoc(cv::abs(cv::Mat_<int>(gl) - cv::Mat_<int>(refL)), nullptr, &maxd);
  EXPECT_LE(maxd, 1.0);
  EXPECT_LT(cv::countNonZero(gl != refL), kW * kH / 1000) << "more than 0.1% of CLAHE pixels differ";
  cv::minMaxLoc(cv::abs(cv::Mat_<int>(gr) - cv::Mat_<int>(refR)), nullptr, &maxd);
  EXPECT_LE(maxd, 1.0);
}

TEST(Lcn, MatchesCpuAndNoiseSigma) {
  GpuKltCore core(params());
  const cv::Mat l = texture(kW, kH, 4, 1.5, 12.0);
  cv::Mat noisy = l.clone();
  cv::Mat noise(kH, kW, CV_8S);
  cv::randn(noise, 0, 1.2);
  cv::add(noisy, noise, noisy, cv::noArray(), CV_8U);
  feed(core, noisy, noisy);
  core.frame(true, true, 1.5);
  core.sync();
  const double sigma = cpuNoiseSigma(noisy);
  EXPECT_NEAR(core.lastSigma(), sigma, 1e-9 * std::max(1.0, sigma));
  const cv::Mat ref = cpuLcn(noisy, sigma, 15, 1.5);
  const cv::Mat g = download(core, PyrId::Cur, 0);
  cv::Mat d;
  cv::absdiff(g, ref, d);
  double maxd = 0;
  cv::minMaxLoc(d, nullptr, &maxd);
  EXPECT_LE(maxd, 1.0);
  EXPECT_LT(cv::mean(d)[0], 0.01);
}

TEST(Detect, MinEigAndFeaturesMatchGoodFeaturesToTrack) {
  CoreParams p = params();
  GpuKltCore core(p);
  const cv::Mat l = texture(kW, kH, 5, 1.2, 40.0);
  feed(core, l, l);
  core.frame(false, true, 0.0);
  // Detection runs on the tracking image; use the GPU's own so that CLAHE rounding cannot enter the comparison.
  const cv::Mat cl = download(core, PyrId::Cur, 0);
  // Exclusion mask: a few discs, like tracked features.
  cv::Mat mask(kH, kW, CV_8UC1, core.maskBuffer());
  mask.setTo(255);
  for (const cv::Point c : {cv::Point(100, 100), cv::Point(300, 200), cv::Point(450, 400), cv::Point(91, 91)})
    cv::circle(mask, c, 15, cv::Scalar(0), -1);
  int* need = core.needBuffer();
  const int cells = core.numCells();
  for (int c = 0; c < cells; ++c) need[c] = 3;
  core.detect();
  core.sync();
  int matched = 0, total = 0;
  double maxRel = 0;
  for (int c = 0; c < cells; ++c) {
    int r[4];
    core.cellRect(c, r);
    const cv::Rect cell(r[0], r[1], r[2], r[3]);
    cv::Mat eigRef;
    cv::cornerMinEigenVal(cl(cell), eigRef, 3, 3);
    cv::Mat eig(cell.height, cell.width, CV_32F);
    core.downloadEig(c, eig.ptr<float>());
    double refMax = 0;
    cv::minMaxLoc(eigRef, nullptr, &refMax);
    cv::Mat diff;
    cv::absdiff(eig, eigRef, diff);
    double dmax = 0;
    cv::minMaxLoc(diff, nullptr, &dmax);
    maxRel = std::max(maxRel, dmax / std::max(refMax, 1e-9));
    std::vector<cv::Point2f> found;
    cv::goodFeaturesToTrack(cl(cell), found, 3, p.quality_level, p.min_distance, mask(cell));
    std::vector<cv::Point2f> got;
    for (int k = 0; k < core.detCount()[c]; ++k) got.emplace_back(core.detPts()[c * p.per_cell + k].x - cell.x, core.detPts()[c * p.per_cell + k].y - cell.y);
    total += static_cast<int>(found.size());
    for (size_t k = 0; k < found.size() && k < got.size(); ++k) matched += found[k] == got[k];
    EXPECT_EQ(found.size(), got.size()) << "cell " << c;
  }
  EXPECT_LT(maxRel, 1e-4) << "min eigenvalue map deviates from cv::cornerMinEigenVal on the cell ROI";
  EXPECT_GE(matched, static_cast<int>(0.98 * total)) << matched << " of " << total << " features identical and in order";
}

TEST(Lk, SubpixelShiftAccuracy) {
  GpuKltCore core(params());
  const cv::Mat a = texture(kW, kH, 6, 2.0, 35.0);
  const cv::Point2f shift(3.37f, -2.81f);
  cv::Mat b;
  const cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
  cv::warpAffine(a, b, M, a.size(), cv::INTER_CUBIC, cv::BORDER_REFLECT_101);
  core.loadPyramid(PyrId::Prev, a.data);
  core.loadPyramid(PyrId::Cur, b.data);
  std::vector<Pt> pts;
  for (int y = 60; y < kH - 60; y += 45)
    for (int x = 60; x < kW - 60; x += 45) pts.push_back(Pt{static_cast<float>(x), static_cast<float>(y)});
  core.trackTemporal(pts.data(), static_cast<int>(pts.size()));
  core.sync();
  double sumErr = 0, maxErr = 0;
  int n = 0, fbOk = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    ASSERT_TRUE(core.statusFwd()[i]) << i;
    const double ex = core.fwd()[i].x - (pts[i].x + shift.x), ey = core.fwd()[i].y - (pts[i].y + shift.y);
    const double e = std::hypot(ex, ey);
    sumErr += e;
    maxErr = std::max(maxErr, e);
    ++n;
    const double fb = std::hypot(core.back()[i].x - pts[i].x, core.back()[i].y - pts[i].y);
    fbOk += fb < 0.1;
  }
  std::vector<cv::Point2f> cvPts, cvOut;
  for (const Pt& q : pts) cvPts.emplace_back(q.x, q.y);
  std::vector<uchar> st;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(a, b, cvPts, cvOut, st, err, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));
  double cvSum = 0;
  for (size_t i = 0; i < pts.size(); ++i) cvSum += std::hypot(cvOut[i].x - (pts[i].x + shift.x), cvOut[i].y - (pts[i].y + shift.y));
  cvSum /= n;
  // Ground truth is a bicubic warp, so the LK error floor is a few hundredths of a pixel; GPU must not be worse than cv.
  EXPECT_LT(sumErr / n, std::max(0.08, 1.2 * cvSum)) << "mean subpixel error [px], cv: " << cvSum;
  EXPECT_LT(maxErr, 0.3) << "max subpixel error [px]";
  EXPECT_GE(fbOk, n * 0.95);
}

TEST(Lk, MatchesCvCalcOpticalFlowPyrLK) {
  GpuKltCore core(params());
  const cv::Mat a = texture(kW, kH, 7, 1.5, 35.0);
  cv::Mat b;
  const cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(kW / 2.f, kH / 2.f), 1.5, 1.01);
  cv::warpAffine(a, b, M, a.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  cv::Mat noise(kH, kW, CV_8S);
  cv::randn(noise, 0, 2.0);
  cv::add(b, noise, b, cv::noArray(), CV_8U);
  core.loadPyramid(PyrId::Prev, a.data);
  core.loadPyramid(PyrId::Cur, b.data);
  std::vector<Pt> pts;
  std::vector<cv::Point2f> cvPts;
  cv::RNG rng(11);
  for (int i = 0; i < 400; ++i) {
    const float x = static_cast<float>(rng.uniform(5.0, kW - 5.0)), y = static_cast<float>(rng.uniform(5.0, kH - 5.0));
    pts.push_back(Pt{x, y});
    cvPts.emplace_back(x, y);
  }
  core.trackTemporal(pts.data(), static_cast<int>(pts.size()));
  core.sync();
  std::vector<cv::Point2f> out, back;
  std::vector<uchar> s1, s2;
  std::vector<float> err;
  const cv::TermCriteria crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(a, b, cvPts, out, s1, err, cv::Size(21, 21), 3, crit);
  cv::calcOpticalFlowPyrLK(b, a, out, back, s2, err, cv::Size(21, 21), 3, crit);
  int statusAgree = 0, both = 0, close01 = 0, close05 = 0;
  std::vector<double> diffs;
  for (size_t i = 0; i < pts.size(); ++i) {
    const bool gOk = core.statusFwd()[i] && core.statusBack()[i];
    const bool cOk = s1[i] && s2[i];
    statusAgree += gOk == cOk;
    if (!gOk || !cOk) continue;
    ++both;
    const double d = std::hypot(core.fwd()[i].x - out[i].x, core.fwd()[i].y - out[i].y);
    diffs.push_back(d);
    close01 += d < 0.1;
    close05 += d < 0.5;
    const double db = std::hypot(core.back()[i].x - back[i].x, core.back()[i].y - back[i].y);
    EXPECT_LT(db, 1.0) << "backward point " << i;
  }
  std::sort(diffs.begin(), diffs.end());
  ASSERT_GT(both, 300);
  EXPECT_GE(statusAgree, static_cast<int>(0.98 * pts.size()));
  EXPECT_LT(diffs[diffs.size() / 2], 0.005) << "median |GPU - cv| [px]";
  EXPECT_GE(close01, static_cast<int>(0.97 * both));
  EXPECT_GE(close05, static_cast<int>(0.99 * both));
}

void expectTrackerAgrees(const uw_ct_vio::KltConfig& cfg, bool expectLcn) {
  using namespace uw_ct_vio;
  CameraModel cam;
  cam.distortion = CameraModel::Distortion::Equidistant;
  cam.fx = cam.fy = 460.0;
  cam.cx = 360.0;
  cam.cy = 270.0;
  cam.coeffs = {0.04, 0.17, -0.3, 0.47};
  cam.width = kW;
  cam.height = kH;
  const gtsam::Pose3 T_C0C1(gtsam::Rot3(), gtsam::Point3(0.11, 0, 0));
  KltStereoTracker cpu(cfg, cam, cam, T_C0C1);
  GpuKltStereoTracker gpu(cfg, cam, cam, T_C0C1);
  const cv::Mat world = texture(kW + 200, kH + 200, 8, 2.0, 30.0);
  size_t same = 0, cpuCount = 0, gpuCount = 0;
  std::vector<double> diffs;
  for (int f = 0; f < 8; ++f) {
    const cv::Rect rl(100 + 2 * f, 100 + f, kW, kH), rr(100 + 2 * f - 8, 100 + f, kW, kH);
    const cv::Mat l = world(rl).clone(), r = world(rr).clone();
    const auto oc = cpu.track(l, r), og = gpu.track(l, r);
    if (f < 4) continue;
    std::map<uint64_t, const FeatureObs*> byId;
    for (const auto& o : og) byId[o.track_id] = &o;
    cpuCount += oc.size();
    gpuCount += og.size();
    for (const auto& o : oc) {
      const auto it = byId.find(o.track_id);
      if (it == byId.end()) continue;
      ++same;
      diffs.push_back(cam.fx * (o.left - it->second->left).norm());
      EXPECT_EQ(o.has_right, it->second->has_right);
    }
  }
  EXPECT_EQ(gpu.lastStats().lcn_active, expectLcn);
  EXPECT_GT(cpuCount, 400u);
  EXPECT_GE(static_cast<double>(same), 0.95 * static_cast<double>(cpuCount)) << "CPU ids present in GPU output";
  EXPECT_NEAR(static_cast<double>(gpuCount), static_cast<double>(cpuCount), 0.05 * cpuCount);
  std::sort(diffs.begin(), diffs.end());
  ASSERT_FALSE(diffs.empty());
  EXPECT_LT(diffs[diffs.size() / 2], 0.02) << "median pixel diff";
}

TEST(Tracker, AgreesWithCpuAutoLcnTemporalOnly) {  // --lcn 2 --lcn-stereo 0, switches on at frame 3
  uw_ct_vio::KltConfig cfg;
  cfg.lcn_auto = true;
  cfg.lcn_stereo = false;
  cfg.lcn_auto_frames = 3;
  expectTrackerAgrees(cfg, true);
}

TEST(Tracker, AgreesWithCpuClaheOnly) {
  expectTrackerAgrees(uw_ct_vio::KltConfig(), false);
}

TEST(Tracker, AgreesWithCpuLcnStereo) {
  uw_ct_vio::KltConfig cfg;
  cfg.lcn = true;
  cfg.lcn_stereo = true;
  expectTrackerAgrees(cfg, true);
}

TEST(Tracker, RejectsUnsupportedSwitches) {
  using namespace uw_ct_vio;
  CameraModel cam;
  cam.fx = cam.fy = 460.0;
  cam.cx = 360.0;
  cam.cy = 270.0;
  cam.coeffs = {0, 0, 0, 0};
  cam.width = kW;
  cam.height = kH;
  KltConfig cfg;
  cfg.map_predict = true;
  EXPECT_THROW(GpuKltStereoTracker(cfg, cam, cam, gtsam::Pose3()), std::invalid_argument);
}
