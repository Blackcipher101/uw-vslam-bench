// CUDA kernels of the uw_ct_vio KLT stereo front end behind a plain C++ interface (no CUDA, OpenCV or GTSAM
// types), so that only GpuKltCore.cu is compiled by nvcc.
//
// All work is enqueued on one stream; results are valid after sync(). Host staging buffers are pinned and
// allocated once. Image coordinates are pixels (x right, y down, pixel centres at integer coordinates, as OpenCV).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace gpu_frontend {

struct Pt {
  float x, y;
};

struct CoreParams {
  int width = 0, height = 0;      ///< image size [px]; both cameras
  int lk_window = 21;             ///< LK window side [px]
  int lk_levels = 3;              ///< temporal pyramid levels above level 0 (cv maxLevel)
  int stereo_levels = 4;          ///< stereo pyramid levels above level 0
  int lk_max_iter = 30;
  double lk_eps = 0.01;           ///< LK convergence [px]
  double lk_min_eig = 1e-4;       ///< cv minEigThreshold
  int max_points = 1024;          ///< LK capacity
  int grid_cols = 8, grid_rows = 6;
  int per_cell = 3;               ///< upper bound of new features per grid cell
  int min_distance = 15;          ///< [px]
  double quality_level = 0.001;
  double clahe_clip = 1.0;
  int clahe_tiles = 2;
  int lcn_window = 15;            ///< [px]
};

/// Which pyramid a call refers to. Prev/Cur are the temporal ping-pong pair of left tracking images.
enum class PyrId { Prev, Cur, Right, ClaheLeft };

/// Stream timestamps for per-stage GPU timing (a Start/Done pair brackets a stage without the host gaps).
enum class Mark { FrameStart = 0, PreDone, TemporalDone, DetectStart, DetectDone, StereoStart, StereoDone, Count };

class GpuKltCore {
 public:
  explicit GpuKltCore(const CoreParams& p);  ///< throws std::runtime_error on CUDA failure
  ~GpuKltCore();
  GpuKltCore(const GpuKltCore&) = delete;
  GpuKltCore& operator=(const GpuKltCore&) = delete;

  const CoreParams& params() const;
  int numCells() const;
  /// Cell rectangle {x, y, w, h} exactly as KltStereoTracker::detect computes it.
  void cellRect(int cell, int* rect4) const;

  // ---- Pinned staging buffers (width*height bytes, contiguous). The accessors wait for the previous upload. ----
  uint8_t* leftBuffer();
  uint8_t* rightBuffer();
  uint8_t* maskBuffer();  ///< 255 = free, 0 = excluded (uploaded by detect)
  int* needBuffer();      ///< numCells() entries: new features wanted per cell (uploaded by detect)

  /// Enqueue the upload of leftBuffer()/rightBuffer().
  void upload();
  /// Build the tracking pyramids of the uploaded pair (the previous Cur becomes Prev), and the Sobel gradients
  /// the detector needs. lcn: left is locally normalised (right too if lcn_stereo), else CLAHE; with lcn and
  /// !lcn_stereo the stereo pair is CLAHE.
  void frame(bool lcn, bool lcn_stereo, double lcn_min_floor);
  /// Replace Cur (the previous frame's tracking image) by its LCN version computed from a raw image
  /// (the lcn_auto switch-over). Uses the left staging buffer as scratch, so call before filling it.
  void reseedCurAsLcn(const uint8_t* raw, double lcn_min_floor);

  // ---- LK. lkPair tracks pts (level-0 coordinates of `from`) into `to`, then back again starting at the forward
  // result, like the CPU forward-backward check. Results via fwd()/back()/statusFwd()/statusBack() after sync().
  void lkPair(PyrId from, PyrId to, int levels, const Pt* pts, int n);
  /// Temporal pair: Prev -> Cur with lk_levels.
  void trackTemporal(const Pt* pts, int n) { lkPair(PyrId::Prev, PyrId::Cur, params().lk_levels, pts, n); }
  /// Stereo pair: left -> right with stereo_levels (left is CLAHE left if lcn && !lcn_stereo).
  void trackStereo(const Pt* pts, int n);
  const Pt* fwd() const;
  const Pt* back() const;
  const uint8_t* statusFwd() const;
  const uint8_t* statusBack() const;

  /// Shi-Tomasi per grid cell with needBuffer()/maskBuffer(); results via detCount()/detPts() after sync().
  void detect();
  const int* detCount() const;  ///< per cell
  const Pt* detPts() const;     ///< [cell * per_cell + k]

  void sync();
  /// Benchmark aid for GPUs with dynamic clocks (laptops): run a one-warp spin kernel on a second stream so the
  /// GPU stays at its busy clock between the short per-frame bursts. Not for production use.
  void startClockKeepAlive();
  void stopClockKeepAlive();
  void mark(Mark m);
  float elapsedMs(Mark a, Mark b);  ///< after sync()
  double lastSigma() const;         ///< LCN noise sigma of the last frame() with lcn, valid after sync()

  // ---- Test / debugging hooks (synchronous) ----
  /// Load a left-size image as level 0 of a pyramid and build its levels and derivatives.
  void loadPyramid(PyrId id, const uint8_t* img);
  void levelSize(PyrId id, int level, int* w, int* h) const;
  int levelCount(PyrId id) const;
  void downloadLevel(PyrId id, int level, uint8_t* dst);
  void downloadDeriv(PyrId id, int level, int16_t* dst_xy);  ///< interleaved dx, dy
  void downloadEig(int cell, float* dst);                    ///< cell-local min eigenvalue map from the last detect()

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu_frontend
