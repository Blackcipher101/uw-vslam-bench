// GPU implementation of uw_ct_vio::KltStereoTracker (same configuration, same output type): LCN/CLAHE, pyramids,
// Shi-Tomasi detection, temporal and stereo LK with forward-backward checks run in CUDA; undistortion, the
// essential-matrix RANSAC and the epipolar test stay on the CPU (few hundred points).
//
// Supported KltConfig: everything of the plain tracker except map_predict, reacquire, snr_sigma and
// ncc_noise_aware (the constructor throws std::invalid_argument for those). `pred` is ignored.
#pragma once

#include <memory>
#include <vector>

#include "gpu_frontend/GpuKltCore.hpp"
#include "uw_ct_vio/KltStereoTracker.hpp"

namespace gpu_frontend {

class GpuKltStereoTracker {
 public:
  /// T_C0C1: right camera pose in the left camera frame (as KltStereoTracker).
  GpuKltStereoTracker(const uw_ct_vio::KltConfig& cfg, const uw_ct_vio::CameraModel& left,
                      const uw_ct_vio::CameraModel& right, const gtsam::Pose3& T_C0C1);

  /// Grey 8-bit images of the same stereo frame, equal to the camera size. Returns features in normalised coordinates.
  std::vector<uw_ct_vio::FeatureObs> track(const cv::Mat& left, const cv::Mat& right,
                                           const uw_ct_vio::MapPrediction* pred = nullptr);

  const uw_ct_vio::KltStereoTracker::Stats& lastStats() const { return stats_; }

  /// Benchmark hook: replace the tracked set (pixel positions in the left image, ids) and the next id to hand out,
  /// e.g. to re-synchronise with the CPU tracker every frame for open-loop (one-frame) agreement measurements.
  void setTracks(const std::vector<cv::Point2f>& px, const std::vector<uint64_t>& ids, uint64_t nextId);

  /// Per-frame stage times [ms]. GPU stages are stream-event times of the kernels and copies of that stage; host
  /// stages are wall-clock. A stage that did not run in the frame is 0.
  struct Timing {
    double gpu_pre_ms = 0;       ///< image upload, noise sigma, LCN/CLAHE, pyramids, derivatives, Sobel
    double gpu_temporal_ms = 0;  ///< temporal LK forward + backward (incl. point transfer)
    double host_ransac_ms = 0;   ///< FB/NCC filtering, undistortion, findEssentialMat
    double gpu_detect_ms = 0;    ///< mask upload, cell eigenvalues, selection, result download
    double gpu_stereo_ms = 0;    ///< stereo LK forward + backward
    double host_post_ms = 0;     ///< undistortion of the stereo points and the epipolar test
    double total_ms = 0;         ///< wall clock of track()
  };
  const Timing& lastTiming() const { return timing_; }
  GpuKltCore& core() { return core_; }

 private:
  uw_ct_vio::KltConfig cfg_;
  uw_ct_vio::CameraModel left_, right_;
  cv::Matx33d E_;
  GpuKltCore core_;
  cv::Mat prevRaw_;
  bool havePrev_ = false;
  bool lcnActive_ = false, lcnDecided_ = false;
  std::vector<double> autoRanges_;
  double autoRange_ = -1.0;
  std::vector<cv::Point2f> pts_;
  std::vector<uint64_t> ids_;
  uint64_t nextId_ = 1;
  uw_ct_vio::KltStereoTracker::Stats stats_;
  Timing timing_;
};

}  // namespace gpu_frontend
