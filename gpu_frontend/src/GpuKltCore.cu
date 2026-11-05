#include "gpu_frontend/GpuKltCore.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_frontend {

namespace {

constexpr int kMaxLevels = 6;           // pyramid levels 0..5
constexpr int kLkMaxPerLane = 17;       // window up to 23x23 = 529 samples <= 32 * 17
constexpr int kWBits = 14;              // cv::calcOpticalFlowPyrLK fixed-point bilinear weights
constexpr float kLkFltScale = 1.f / (1 << 20);
constexpr double kLcnNoiseFloor = 3.0;  // in units of the image noise sigma (KltStereoTracker.cpp)
constexpr float kLcnScale = 40.f;       // z-score -> 8-bit, centred on 128
constexpr int kSobelKeyBits = 21;       // gx^2 + gy^2 <= 2 * 1020^2 < 2^21

#define CUDA_CHECK(call)                                                                              \
  do {                                                                                                \
    const cudaError_t e_ = (call);                                                                    \
    if (e_ != cudaSuccess)                                                                            \
      throw std::runtime_error(std::string("CUDA error '") + cudaGetErrorString(e_) + "' at " +       \
                               __FILE__ + ":" + std::to_string(__LINE__) + " in " #call);             \
  } while (0)

struct PyrView {
  const uint8_t* img[kMaxLevels];
  const short2* der[kMaxLevels];
  int w[kMaxLevels], h[kMaxLevels];
};

struct LkParams {
  int maxLevel, win, maxIter, useInit;
  float eps2, minEig;
};

struct CellRect {
  int x, y, w, h;
};

__device__ __forceinline__ int refl(int i, int n) {  // BORDER_REFLECT_101
  if (n == 1) return 0;
  if (i < 0) i = -i;
  if (i >= n) i = 2 * (n - 1) - i;
  return min(max(i, 0), n - 1);
}

__device__ __forceinline__ int satU8(float v) { return min(max(__float2int_rn(v), 0), 255); }

// ------------------------------------------------------------------------------------------------ pyramid

// cv::pyrDown (5-tap binomial, reflect-101, (sum + 128) >> 8): bit exact for 8-bit images.
__global__ void k_pyrdown(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  const int wt[5] = {1, 4, 6, 4, 1};
  int sum = 0;
  for (int j = 0; j < 5; ++j) {
    const uint8_t* row = src + refl(2 * y + j - 2, sh) * sw;
    int r = 0;
    for (int i = 0; i < 5; ++i) r += wt[i] * row[refl(2 * x + i - 2, sw)];
    sum += wt[j] * r;
  }
  dst[y * dw + x] = static_cast<uint8_t>((sum + 128) >> 8);
}

// cv calcScharrDeriv: dx = [-1 0 1] * [3 10 3]^T, dy = [3 10 3] * [-1 0 1]^T, reflect-101, 16-bit.
__global__ void k_scharr(const uint8_t* img, int w, int h, short2* der) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int ym = refl(y - 1, h), yp = refl(y + 1, h);
  int dx = 0, dy = 0;
  for (int k = -1; k <= 1; k += 2) {
    const int xx = refl(x + k, w);
    dx += k * (3 * (img[ym * w + xx] + img[yp * w + xx]) + 10 * img[y * w + xx]);
  }
  const int xm = refl(x - 1, w), xp = refl(x + 1, w);
  const int tm = img[yp * w + xm] - img[ym * w + xm], t0 = img[yp * w + x] - img[ym * w + x],
            tp = img[yp * w + xp] - img[ym * w + xp];
  dy = 3 * (tm + tp) + 10 * t0;
  der[y * w + x] = make_short2(static_cast<short>(dx), static_cast<short>(dy));
}

// ------------------------------------------------------------------------------------------------ LK

__device__ __forceinline__ float warpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct BilW {
  int w00, w01, w10, w11;
};

__device__ __forceinline__ BilW bilWeights(float a, float b) {
  BilW r;
  r.w00 = __float2int_rn(__fmul_rn(__fmul_rn(1.f - a, 1.f - b), static_cast<float>(1 << kWBits)));
  r.w01 = __float2int_rn(__fmul_rn(__fmul_rn(a, 1.f - b), static_cast<float>(1 << kWBits)));
  r.w10 = __float2int_rn(__fmul_rn(__fmul_rn(1.f - a, b), static_cast<float>(1 << kWBits)));
  r.w11 = (1 << kWBits) - r.w00 - r.w01 - r.w10;
  return r;
}

__device__ __forceinline__ int ldImg(const uint8_t* img, int w, int h, int x, int y) {
  return img[refl(y, h) * w + refl(x, w)];
}

__device__ __forceinline__ short2 ldDer(const short2* d, int w, int h, int x, int y) {
  return (x < 0 || y < 0 || x >= w || y >= h) ? make_short2(0, 0) : d[y * w + x];
}

// One warp per point, all pyramid levels and iterations in the kernel. Mirrors cv::calcOpticalFlowPyrLK for
// 8-bit images (fixed-point bilinear sampling, Scharr derivatives, minEig test, the same stopping rules).
__global__ void k_lk(PyrView I, PyrView J, const float2* prevPts, float2* nextPts, uint8_t* status, int n, LkParams p) {
  const int lane = threadIdx.x & 31;
  const int pt = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (pt >= n) return;
  const float2 p0 = prevPts[pt];
  const float2 guess = p.useInit ? nextPts[pt] : p0;
  const int win = p.win, ns = win * win;
  const float hw = (win - 1) * 0.5f;
  bool ok = true;
  float2 res = guess;

  for (int lv = p.maxLevel; lv >= 0; --lv) {
    const float sc = static_cast<float>(1.0 / (1 << lv));
    float2 prevPt = make_float2(p0.x * sc, p0.y * sc);
    float2 nextPt;
    if (lv == p.maxLevel) nextPt = p.useInit ? make_float2(guess.x * sc, guess.y * sc) : prevPt;
    else nextPt = make_float2(res.x * 2.f, res.y * 2.f);
    res = nextPt;

    prevPt.x -= hw;
    prevPt.y -= hw;
    const int ix = static_cast<int>(floorf(prevPt.x)), iy = static_cast<int>(floorf(prevPt.y));
    const int W = I.w[lv], H = I.h[lv];
    if (ix < -win || ix >= W || iy < -win || iy >= H) {
      if (lv == 0) ok = false;
      continue;
    }
    const BilW wi = bilWeights(prevPt.x - ix, prevPt.y - iy);
    const uint8_t* Ii = I.img[lv];
    const short2* Id = I.der[lv];
    int Iv[kLkMaxPerLane], Ixv[kLkMaxPerLane], Iyv[kLkMaxPerLane];
    float a11 = 0.f, a12 = 0.f, a22 = 0.f;
#pragma unroll
    for (int k = 0; k < kLkMaxPerLane; ++k) {
      const int s = lane + 32 * k;
      Iv[k] = Ixv[k] = Iyv[k] = 0;
      if (s >= ns) continue;
      const int x = ix + s % win, y = iy + s / win;
      const int v = ldImg(Ii, W, H, x, y) * wi.w00 + ldImg(Ii, W, H, x + 1, y) * wi.w01 +
                    ldImg(Ii, W, H, x, y + 1) * wi.w10 + ldImg(Ii, W, H, x + 1, y + 1) * wi.w11;
      Iv[k] = (v + (1 << (kWBits - 5 - 1))) >> (kWBits - 5);
      const short2 d00 = ldDer(Id, W, H, x, y), d01 = ldDer(Id, W, H, x + 1, y), d10 = ldDer(Id, W, H, x, y + 1),
                   d11 = ldDer(Id, W, H, x + 1, y + 1);
      const int dxs = d00.x * wi.w00 + d01.x * wi.w01 + d10.x * wi.w10 + d11.x * wi.w11;
      const int dys = d00.y * wi.w00 + d01.y * wi.w01 + d10.y * wi.w10 + d11.y * wi.w11;
      Ixv[k] = (dxs + (1 << (kWBits - 1))) >> kWBits;
      Iyv[k] = (dys + (1 << (kWBits - 1))) >> kWBits;
      a11 += static_cast<float>(Ixv[k] * Ixv[k]);
      a12 += static_cast<float>(Ixv[k] * Iyv[k]);
      a22 += static_cast<float>(Iyv[k] * Iyv[k]);
    }
    a11 = warpSum(a11) * kLkFltScale;
    a12 = warpSum(a12) * kLkFltScale;
    a22 = warpSum(a22) * kLkFltScale;
    float D = __fsub_rn(__fmul_rn(a11, a22), __fmul_rn(a12, a12));
    const float dd = a11 - a22;
    const float minEig = (a22 + a11 - sqrtf(__fadd_rn(__fmul_rn(dd, dd), __fmul_rn(4.f * a12, a12)))) / (2.f * ns);
    if (minEig < p.minEig || D < FLT_EPSILON) {
      if (lv == 0) ok = false;
      continue;
    }
    D = 1.f / D;

    nextPt.x -= hw;
    nextPt.y -= hw;
    float2 prevDelta = make_float2(0.f, 0.f);
    const uint8_t* Ji = J.img[lv];
    const int JW = J.w[lv], JH = J.h[lv];
    for (int j = 0; j < p.maxIter; ++j) {
      const int jx = static_cast<int>(floorf(nextPt.x)), jy = static_cast<int>(floorf(nextPt.y));
      if (jx < -win || jx >= JW || jy < -win || jy >= JH) {
        if (lv == 0) ok = false;
        break;
      }
      const BilW wj = bilWeights(nextPt.x - jx, nextPt.y - jy);
      float b1 = 0.f, b2 = 0.f;
#pragma unroll
      for (int k = 0; k < kLkMaxPerLane; ++k) {
        const int s = lane + 32 * k;
        if (s >= ns) continue;
        const int x = jx + s % win, y = jy + s / win;
        const int v = ldImg(Ji, JW, JH, x, y) * wj.w00 + ldImg(Ji, JW, JH, x + 1, y) * wj.w01 +
                      ldImg(Ji, JW, JH, x, y + 1) * wj.w10 + ldImg(Ji, JW, JH, x + 1, y + 1) * wj.w11;
        const int diff = ((v + (1 << (kWBits - 5 - 1))) >> (kWBits - 5)) - Iv[k];
        b1 += static_cast<float>(diff * Ixv[k]);
        b2 += static_cast<float>(diff * Iyv[k]);
      }
      b1 = warpSum(b1) * kLkFltScale;
      b2 = warpSum(b2) * kLkFltScale;
      const float2 delta = make_float2(__fmul_rn(__fsub_rn(__fmul_rn(a12, b2), __fmul_rn(a22, b1)), D),
                                       __fmul_rn(__fsub_rn(__fmul_rn(a12, b1), __fmul_rn(a11, b2)), D));
      nextPt.x += delta.x;
      nextPt.y += delta.y;
      res = make_float2(nextPt.x + hw, nextPt.y + hw);
      if (delta.x * delta.x + delta.y * delta.y <= p.eps2) break;
      if (j > 0 && fabsf(delta.x + prevDelta.x) < 0.01f && fabsf(delta.y + prevDelta.y) < 0.01f) {
        res.x -= delta.x * 0.5f;
        res.y -= delta.y * 0.5f;
        break;
      }
      prevDelta = delta;
    }
  }
  if (lane == 0) {
    nextPts[pt] = res;
    status[pt] = ok ? 1 : 0;
  }
}

// ------------------------------------------------------------------------------------------------ Sobel, noise, LCN

// 3x3 Sobel on 8-bit input (reflect-101): exact integer gx, gy.
__global__ void k_sobel(const uint8_t* img, int w, int h, short2* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int xm = refl(x - 1, w), xp = refl(x + 1, w);
  int gx = 0, gy = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    const uint8_t* row = img + refl(y + dy, h) * w;
    const int wy = dy == 0 ? 2 : 1;
    gx += wy * (row[xp] - row[xm]);
  }
  const uint8_t* rm = img + refl(y - 1, h) * w;
  const uint8_t* rp = img + refl(y + 1, h) * w;
  gy = (rp[xm] + 2 * rp[x] + rp[xp]) - (rm[xm] + 2 * rm[x] + rm[xp]);
  out[y * w + x] = make_short2(static_cast<short>(gx), static_cast<short>(gy));
}

// Immerkaer noise estimator pieces: ordering key of the gradient magnitude (monotone in gx^2 + gy^2), and the
// absolute response of the 3x3 [1 -2 1; -2 4 -2; 1 -2 1] kernel on interior pixels.
__global__ void k_noise_keys(const uint8_t* img, const short2* sob, int w, int h, uint32_t* keys, int* absResp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const short2 g = sob[y * w + x];
  keys[y * w + x] = static_cast<uint32_t>(g.x * g.x + g.y * g.y);
  int r = 0;
  if (x >= 1 && y >= 1 && x < w - 1 && y < h - 1) {
    const uint8_t* c = img + y * w + x;
    r = c[-w - 1] + c[-w + 1] + c[w - 1] + c[w + 1] - 2 * (c[-w] + c[-1] + c[1] + c[w]) + 4 * c[0];
    r = r < 0 ? -r : r;
  }
  absResp[y * w + x] = r;
}

__global__ void k_noise_sum(const uint32_t* sortedKeys, int n, const uint32_t* keys, const int* absResp, int w, int h,
                            unsigned long long* acc) {
  const uint32_t med = sortedKeys[n / 2];
  unsigned long long s = 0, c = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int x = i % w, y = i / w;
    if (x >= 1 && y >= 1 && x < w - 1 && y < h - 1 && keys[i] <= med) {
      s += static_cast<unsigned long long>(absResp[i]);
      ++c;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
    c += __shfl_xor_sync(0xffffffffu, c, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&acc[0], s);
    atomicAdd(&acc[1], c);
  }
}

__global__ void k_sigma(const unsigned long long* acc, double minFloor, double* sigma, double* floor2) {
  const double sum = static_cast<double>(acc[0]);
  const double n = static_cast<double>(acc[1]);
  const double sg = acc[1] ? std::sqrt(M_PI / 2.0) / 6.0 * sum / n : 0.0;
  const double f = fmax(kLcnNoiseFloor * fmax(sg, 0.5), minFloor);
  *sigma = sg;
  *floor2 = f * f;
}

__global__ void k_box_h(const uint8_t* img, int w, int h, int k, int* s1, int* s2) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uint8_t* row = img + y * w;
  const int r = k / 2;
  int a = 0, b = 0;
  for (int i = -r; i <= r; ++i) {
    const int v = row[refl(x + i, w)];
    a += v;
    b += v * v;
  }
  s1[y * w + x] = a;
  s2[y * w + x] = b;
}

// Local contrast normalisation z = (I - mean) / sqrt(max(var + floor2, floor2)), 8-bit = 128 + 40 z. Float
// operation order follows KltStereoTracker::trackingImage (separate roundings, no FMA contraction).
__global__ void k_lcn(const uint8_t* img, const int* s1, const int* s2, int w, int h, int k, const double* floor2,
                      uint8_t* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int r = k / 2;
  int a = 0, b = 0;
  for (int j = -r; j <= r; ++j) {
    const int yy = refl(y + j, h);
    a += s1[yy * w + x];
    b += s2[yy * w + x];
  }
  const double scale = 1.0 / (static_cast<double>(k) * k);
  const float mu = static_cast<float>(static_cast<double>(a) * scale);
  const float mu2 = static_cast<float>(static_cast<double>(b) * scale);
  const float f2 = static_cast<float>(*floor2);
  float var = __fadd_rn(__fsub_rn(mu2, __fmul_rn(mu, mu)), f2);
  var = __fsqrt_rn(fmaxf(var, f2));
  const float f = static_cast<float>(img[y * w + x]);
  const float z = __fdiv_rn(__fsub_rn(f, mu), var);
  out[y * w + x] = static_cast<uint8_t>(satU8(fmaf(z, kLcnScale, 128.f)));
}

// ------------------------------------------------------------------------------------------------ CLAHE

// Mirrors cv::CLAHE (apply with tilesX x tilesY tiles that divide the image).
__global__ void k_clahe_hist(const uint8_t* img, int w, int tileW, int tileH, int tilesX, int rowsPerBlock, int* hist) {
  __shared__ int sh[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const int tile = blockIdx.y, tx = tile % tilesX, ty = tile / tilesX;
  const int y0 = blockIdx.x * rowsPerBlock, y1 = min(y0 + rowsPerBlock, tileH);
  for (int idx = threadIdx.x; idx < (y1 - y0) * tileW; idx += blockDim.x) {
    const int y = ty * tileH + y0 + idx / tileW, x = tx * tileW + idx % tileW;
    atomicAdd(&sh[img[y * w + x]], 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&hist[tile * 256 + i], sh[i]);
}

__global__ void k_clahe_lut(const int* hist, int clipLimit, float lutScale, uint8_t* lut) {
  __shared__ int h[256];
  __shared__ int clipped;
  const int t = blockIdx.x, i = threadIdx.x;  // 256 threads
  if (i == 0) clipped = 0;
  h[i] = hist[t * 256 + i];
  __syncthreads();
  if (clipLimit > 0) {
    if (h[i] > clipLimit) atomicAdd(&clipped, h[i] - clipLimit);
    __syncthreads();
    int v = min(h[i], clipLimit);
    const int batch = clipped / 256, residual = clipped - batch * 256;
    v += batch;
    if (residual != 0) {
      const int step = max(256 / residual, 1);
      if (i % step == 0 && i / step < residual) ++v;
    }
    __syncthreads();
    h[i] = v;
  }
  __syncthreads();
  if (i == 0) {
    int sum = 0;
    for (int b = 0; b < 256; ++b) {
      sum += h[b];
      h[b] = sum;
    }
  }
  __syncthreads();
  lut[t * 256 + i] = static_cast<uint8_t>(satU8(static_cast<float>(h[i]) * lutScale));
}

__global__ void k_clahe_apply(const uint8_t* img, int w, int h, int tileW, int tileH, int tilesX, int tilesY,
                              const uint8_t* lut, uint8_t* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float tyf = y * (1.0f / tileH) - 0.5f;
  int ty1 = static_cast<int>(floorf(tyf)), ty2 = ty1 + 1;
  const float ya = tyf - ty1, ya1 = 1.0f - ya;
  ty1 = max(ty1, 0);
  ty2 = min(ty2, tilesY - 1);
  const float txf = x * (1.0f / tileW) - 0.5f;
  int tx1 = static_cast<int>(floorf(txf)), tx2 = tx1 + 1;
  const float xa = txf - tx1, xa1 = 1.0f - xa;
  tx1 = max(tx1, 0);
  tx2 = min(tx2, tilesX - 1);
  const int v = img[y * w + x];
  const float l11 = lut[(ty1 * tilesX + tx1) * 256 + v], l12 = lut[(ty1 * tilesX + tx2) * 256 + v];
  const float l21 = lut[(ty2 * tilesX + tx1) * 256 + v], l22 = lut[(ty2 * tilesX + tx2) * 256 + v];
  const float res = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(l11, xa1), __fmul_rn(l12, xa)), ya1),
                              __fmul_rn(__fadd_rn(__fmul_rn(l21, xa1), __fmul_rn(l22, xa)), ya));
  out[y * w + x] = static_cast<uint8_t>(satU8(res));
}

// ------------------------------------------------------------------------------------------------ detection

// Per needy cell: min eigenvalue of the 3x3-summed gradient covariance (cv::cornerMinEigenVal on the cell ROI:
// gradients see the real neighbours, the summing box reflects at the cell border), and the maximum over the
// unmasked pixels.
__global__ void k_eig(const short2* sob, int W, const uint8_t* mask, const CellRect* rects, const int* need,
                      float* eigBuf, size_t cellStride, float* cellMax, float gscale) {
  const int cell = blockIdx.x;
  if (need[cell] <= 0) return;
  const CellRect r = rects[cell];
  float lmax = -FLT_MAX;
  for (int idx = threadIdx.x; idx < r.w * r.h; idx += blockDim.x) {
    const int lx = idx % r.w, ly = idx / r.w;
    float a = 0.f, b = 0.f, c = 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
      const int yy = r.y + refl(ly + dy, r.h);
      for (int dx = -1; dx <= 1; ++dx) {
        const short2 g = sob[yy * W + r.x + refl(lx + dx, r.w)];
        const float gx = static_cast<float>(g.x) * gscale, gy = static_cast<float>(g.y) * gscale;
        a += gx * gx;
        b += gx * gy;
        c += gy * gy;
      }
    }
    a *= 0.5f;
    c *= 0.5f;
    const float e = (a + c) - sqrtf((a - c) * (a - c) + b * b);
    eigBuf[cell * cellStride + idx] = e;
    if (mask[(r.y + ly) * W + r.x + lx]) lmax = fmaxf(lmax, e);
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lmax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -FLT_MAX;
    for (int i = 0; i < (blockDim.x >> 5); ++i) m = fmaxf(m, red[i]);
    cellMax[cell] = m == -FLT_MAX ? 0.f : m;
  }
}

// cv::goodFeaturesToTrack on one cell: threshold at quality * max, 3x3 non-maximum suppression (ignoring the mask),
// mask test, strongest first (ties: larger offset first), greedy minimum distance, at most need[cell] corners.
__global__ void k_select(const float* eigBuf, size_t cellStride, const uint8_t* mask, int W, const CellRect* rects,
                         const int* need, const float* cellMax, double quality, int minDist, int perCell, Pt* outPts,
                         int* outCount) {
  extern __shared__ float cand[];
  __shared__ float sv[256];
  __shared__ int si[256];
  __shared__ int picked;
  const int cell = blockIdx.x, tid = threadIdx.x;
  if (tid == 0) outCount[cell] = 0;
  if (need[cell] <= 0) return;
  const CellRect r = rects[cell];
  const float* e = eigBuf + cell * cellStride;
  const float thr = static_cast<float>(static_cast<double>(cellMax[cell]) * quality);
  auto T = [&](int x, int y) {
    const float v = e[y * r.w + x];
    return v > thr ? v : 0.f;
  };
  for (int idx = tid; idx < r.w * r.h; idx += blockDim.x) {
    const int lx = idx % r.w, ly = idx / r.w;
    float v = 0.f;
    if (lx >= 1 && ly >= 1 && lx < r.w - 1 && ly < r.h - 1) {
      const float t0 = T(lx, ly);
      if (t0 != 0.f && mask[(r.y + ly) * W + r.x + lx]) {
        float m = t0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) m = fmaxf(m, T(lx + dx, ly + dy));
        if (t0 == m) v = t0;
      }
    }
    cand[idx] = v;
  }
  __syncthreads();
  const float md2 = static_cast<float>(minDist) * static_cast<float>(minDist);
  const int need_n = min(need[cell], perCell);
  for (int k = 0; k < need_n; ++k) {
    float bv = 0.f;
    int bi = -1;
    for (int idx = tid; idx < r.w * r.h; idx += blockDim.x) {
      const float v = cand[idx];
      if (v > 0.f && (v > bv || (v == bv && idx > bi))) {
        bv = v;
        bi = idx;
      }
    }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        const float v2 = sv[tid + s];
        const int i2 = si[tid + s];
        if (v2 > sv[tid] || (v2 == sv[tid] && i2 > si[tid])) {
          sv[tid] = v2;
          si[tid] = i2;
        }
      }
      __syncthreads();
    }
    if (tid == 0) picked = si[0];
    __syncthreads();
    const int pk = picked;
    if (pk < 0) break;
    const int px = pk % r.w, py = pk / r.w;
    if (tid == 0) {
      outPts[cell * perCell + k] = Pt{static_cast<float>(r.x + px), static_cast<float>(r.y + py)};
      outCount[cell] = k + 1;
    }
    for (int idx = tid; idx < r.w * r.h; idx += blockDim.x) {
      const float dx = static_cast<float>(idx % r.w - px), dy = static_cast<float>(idx / r.w - py);
      if (dx * dx + dy * dy < md2) cand[idx] = 0.f;
    }
    __syncthreads();
  }
}

__global__ void k_spin(volatile int* stop) {
  float x = 1.f;
  while (*stop == 0) {
    for (int i = 0; i < 4096; ++i) x = fmaf(x, 1.0000001f, 1e-7f);
  }
  if (x == 0.f) *const_cast<int*>(stop) = 2;  // keeps the loop from being optimised away
}

}  // namespace

// ================================================================================================ host side

struct GpuKltCore::Impl {
  CoreParams p;
  int W = 0, H = 0, nPix = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t uploadDone = nullptr;
  cudaEvent_t ev[static_cast<int>(Mark::Count)] = {};

  struct Pyr {
    int levels = 0;
    int w[kMaxLevels] = {}, h[kMaxLevels] = {};
    uint8_t* img[kMaxLevels] = {};
    short2* der[kMaxLevels] = {};
  };
  Pyr P[2], PR, PCL;
  int cur = 0;
  bool stereoUsesClahe = false;

  uint8_t *dRawL = nullptr, *dRawR = nullptr, *dMask = nullptr;
  uint8_t *hLeft = nullptr, *hRight = nullptr, *hMask = nullptr;
  int *hNeed = nullptr, *dNeed = nullptr;
  short2* dSobel = nullptr;
  // noise / LCN
  uint32_t *dKeys = nullptr, *dKeysSorted = nullptr;
  int *dAbsResp = nullptr, *dS1 = nullptr, *dS2 = nullptr;
  unsigned long long* dAcc = nullptr;
  double *dSigma = nullptr, *dFloor2 = nullptr, *hSigma = nullptr;
  void* dSortTmp = nullptr;
  size_t sortTmpBytes = 0;
  // CLAHE
  int* dHist = nullptr;
  uint8_t* dLut = nullptr;
  // LK
  float2 *dPtsIn = nullptr, *dFwd = nullptr, *dBack = nullptr;
  uint8_t *dSFwd = nullptr, *dSBack = nullptr;
  Pt *hPtsIn = nullptr, *hFwd = nullptr, *hBack = nullptr;
  uint8_t *hSFwd = nullptr, *hSBack = nullptr;
  // detection
  std::vector<CellRect> rects;
  CellRect* dRects = nullptr;
  float *dEig = nullptr, *dCellMax = nullptr;
  size_t cellStride = 0;
  Pt* dDetPts = nullptr;
  int* dDetCount = nullptr;
  Pt* hDetPts = nullptr;
  int* hDetCount = nullptr;
  size_t selectSmem = 0;
  cudaStream_t spinStream = nullptr;
  int* spinFlag = nullptr;  // mapped pinned host memory

  template <typename T>
  T* dalloc(size_t n) {
    void* q = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n * sizeof(T)));
    return static_cast<T*>(q);
  }
  template <typename T>
  T* halloc(size_t n) {
    void* q = nullptr;
    CUDA_CHECK(cudaMallocHost(&q, n * sizeof(T)));
    return static_cast<T*>(q);
  }

  void initPyr(Pyr& py, int requestedLevels) {
    py.levels = 1;
    py.w[0] = W;
    py.h[0] = H;
    // cv::buildOpticalFlowPyramid stops once a level is not larger than the window.
    for (int l = 1; l <= requestedLevels && l < kMaxLevels; ++l) {
      const int w = (py.w[l - 1] + 1) / 2, h = (py.h[l - 1] + 1) / 2;
      if (w <= p.lk_window || h <= p.lk_window) break;
      py.w[l] = w;
      py.h[l] = h;
      py.levels = l + 1;
    }
    for (int l = 0; l < py.levels; ++l) {
      py.img[l] = dalloc<uint8_t>(static_cast<size_t>(py.w[l]) * py.h[l]);
      py.der[l] = dalloc<short2>(static_cast<size_t>(py.w[l]) * py.h[l]);
    }
  }
  void freePyr(Pyr& py) {
    for (int l = 0; l < py.levels; ++l) {
      cudaFree(py.img[l]);
      cudaFree(py.der[l]);
    }
    py = Pyr();
  }

  Pyr& pyr(PyrId id) {
    switch (id) {
      case PyrId::Prev: return P[1 - cur];
      case PyrId::Cur: return P[cur];
      case PyrId::Right: return PR;
      case PyrId::ClaheLeft: return PCL;
    }
    return P[cur];
  }
  const Pyr& pyr(PyrId id) const { return const_cast<Impl*>(this)->pyr(id); }

  PyrView view(const Pyr& py) const {
    PyrView v{};
    for (int l = 0; l < py.levels; ++l) {
      v.img[l] = py.img[l];
      v.der[l] = py.der[l];
      v.w[l] = py.w[l];
      v.h[l] = py.h[l];
    }
    return v;
  }

  void buildLevels(const Pyr& py) {
    const dim3 blk(32, 8);
    for (int l = 1; l < py.levels; ++l)
      k_pyrdown<<<dim3((py.w[l] + 31) / 32, (py.h[l] + 7) / 8), blk, 0, stream>>>(py.img[l - 1], py.w[l - 1], py.h[l - 1],
                                                                                  py.img[l], py.w[l], py.h[l]);
    for (int l = 0; l < py.levels; ++l)
      k_scharr<<<dim3((py.w[l] + 31) / 32, (py.h[l] + 7) / 8), blk, 0, stream>>>(py.img[l], py.w[l], py.h[l], py.der[l]);
    CUDA_CHECK(cudaGetLastError());
  }

  dim3 grid2d() const { return dim3((W + 31) / 32, (H + 7) / 8); }

  void sobel(const uint8_t* img) {
    k_sobel<<<grid2d(), dim3(32, 8), 0, stream>>>(img, W, H, dSobel);
    CUDA_CHECK(cudaGetLastError());
  }

  void noiseSigma(const uint8_t* raw, double minFloor) {
    sobel(raw);
    k_noise_keys<<<grid2d(), dim3(32, 8), 0, stream>>>(raw, dSobel, W, H, dKeys, dAbsResp);
    size_t bytes = sortTmpBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(dSortTmp, bytes, dKeys, dKeysSorted, nPix, 0, kSobelKeyBits, stream));
    CUDA_CHECK(cudaMemsetAsync(dAcc, 0, 2 * sizeof(unsigned long long), stream));
    k_noise_sum<<<64, 256, 0, stream>>>(dKeysSorted, nPix, dKeys, dAbsResp, W, H, dAcc);
    k_sigma<<<1, 1, 0, stream>>>(dAcc, minFloor, dSigma, dFloor2);
    CUDA_CHECK(cudaMemcpyAsync(hSigma, dSigma, sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaGetLastError());
  }

  void lcn(const uint8_t* raw, uint8_t* out) {
    k_box_h<<<grid2d(), dim3(32, 8), 0, stream>>>(raw, W, H, p.lcn_window, dS1, dS2);
    k_lcn<<<grid2d(), dim3(32, 8), 0, stream>>>(raw, dS1, dS2, W, H, p.lcn_window, dFloor2, out);
    CUDA_CHECK(cudaGetLastError());
  }

  void clahe(const uint8_t* raw, uint8_t* out) {
    const int tiles = p.clahe_tiles * p.clahe_tiles, tw = W / p.clahe_tiles, th = H / p.clahe_tiles;
    const int tileTotal = tw * th;
    int clip = 0;
    if (p.clahe_clip > 0.0) clip = std::max(static_cast<int>(p.clahe_clip * tileTotal / 256), 1);
    const int strips = 16, rowsPer = (th + strips - 1) / strips;
    CUDA_CHECK(cudaMemsetAsync(dHist, 0, tiles * 256 * sizeof(int), stream));
    k_clahe_hist<<<dim3(strips, tiles), 256, 0, stream>>>(raw, W, tw, th, p.clahe_tiles, rowsPer, dHist);
    k_clahe_lut<<<tiles, 256, 0, stream>>>(dHist, clip, static_cast<float>(255) / tileTotal, dLut);
    k_clahe_apply<<<grid2d(), dim3(32, 8), 0, stream>>>(raw, W, H, tw, th, p.clahe_tiles, p.clahe_tiles, dLut, out);
    CUDA_CHECK(cudaGetLastError());
  }

  void waitUpload() { CUDA_CHECK(cudaEventSynchronize(uploadDone)); }

  void lk(const Pyr& from, const Pyr& to, int levels, int n) {
    if (n < 0 || n > p.max_points) throw std::invalid_argument("GpuKltCore::lk: point count exceeds max_points");
    CUDA_CHECK(cudaMemcpyAsync(dPtsIn, hPtsIn, n * sizeof(Pt), cudaMemcpyHostToDevice, stream));
    if (n == 0) return;
    const int lv = std::min({levels, from.levels - 1, to.levels - 1});
    LkParams lp{lv, p.lk_window, p.lk_max_iter, 0, static_cast<float>(p.lk_eps * p.lk_eps), static_cast<float>(p.lk_min_eig)};
    const int warps = 4, blocks = (n + warps - 1) / warps;
    k_lk<<<blocks, warps * 32, 0, stream>>>(view(from), view(to), dPtsIn, dFwd, dSFwd, n, lp);
    // Backward from the forward result: nextPts initialised to the forward points (no initial-flow flag in cv).
    k_lk<<<blocks, warps * 32, 0, stream>>>(view(to), view(from), dFwd, dBack, dSBack, n, lp);
    CUDA_CHECK(cudaMemcpyAsync(hFwd, dFwd, n * sizeof(Pt), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hBack, dBack, n * sizeof(Pt), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hSFwd, dSFwd, n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hSBack, dSBack, n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaGetLastError());
  }
};

GpuKltCore::GpuKltCore(const CoreParams& p) : impl_(new Impl) {
  Impl& s = *impl_;
  s.p = p;
  if (p.width <= 0 || p.height <= 0) throw std::invalid_argument("GpuKltCore: invalid image size");
  if (p.lk_window % 2 == 0 || p.lk_window * p.lk_window > 32 * kLkMaxPerLane || p.lk_window < 3)
    throw std::invalid_argument("GpuKltCore: lk_window must be odd, 3..23");
  if (p.lk_levels >= kMaxLevels || p.stereo_levels >= kMaxLevels)
    throw std::invalid_argument("GpuKltCore: at most 5 pyramid levels above level 0");
  if (p.width % p.clahe_tiles != 0 || p.height % p.clahe_tiles != 0)
    throw std::invalid_argument("GpuKltCore: image size must be divisible by clahe_tiles");
  if (p.lcn_window % 2 == 0) throw std::invalid_argument("GpuKltCore: lcn_window must be odd");
  s.W = p.width;
  s.H = p.height;
  s.nPix = s.W * s.H;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&s.uploadDone, cudaEventDisableTiming));
  for (auto& e : s.ev) CUDA_CHECK(cudaEventCreate(&e));
  const int levels = std::max(p.lk_levels, p.stereo_levels);
  s.initPyr(s.P[0], levels);
  s.initPyr(s.P[1], levels);
  s.initPyr(s.PR, levels);
  s.initPyr(s.PCL, levels);
  const size_t n = static_cast<size_t>(s.nPix);
  s.dRawL = s.dalloc<uint8_t>(n);
  s.dRawR = s.dalloc<uint8_t>(n);
  s.dMask = s.dalloc<uint8_t>(n);
  s.hLeft = s.halloc<uint8_t>(n);
  s.hRight = s.halloc<uint8_t>(n);
  s.hMask = s.halloc<uint8_t>(n);
  const int cells = p.grid_cols * p.grid_rows;
  s.hNeed = s.halloc<int>(cells);
  s.dNeed = s.dalloc<int>(cells);
  std::memset(s.hNeed, 0, cells * sizeof(int));
  s.dSobel = s.dalloc<short2>(n);
  s.dKeys = s.dalloc<uint32_t>(n);
  s.dKeysSorted = s.dalloc<uint32_t>(n);
  s.dAbsResp = s.dalloc<int>(n);
  s.dS1 = s.dalloc<int>(n);
  s.dS2 = s.dalloc<int>(n);
  s.dAcc = s.dalloc<unsigned long long>(2);
  s.dSigma = s.dalloc<double>(1);
  s.dFloor2 = s.dalloc<double>(1);
  s.hSigma = s.halloc<double>(1);
  *s.hSigma = 0.0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, s.sortTmpBytes, s.dKeys, s.dKeysSorted, s.nPix, 0, kSobelKeyBits));
  s.dSortTmp = s.dalloc<uint8_t>(s.sortTmpBytes);
  s.dHist = s.dalloc<int>(p.clahe_tiles * p.clahe_tiles * 256);
  s.dLut = s.dalloc<uint8_t>(p.clahe_tiles * p.clahe_tiles * 256);
  const size_t m = static_cast<size_t>(p.max_points);
  s.dPtsIn = s.dalloc<float2>(m);
  s.dFwd = s.dalloc<float2>(m);
  s.dBack = s.dalloc<float2>(m);
  s.dSFwd = s.dalloc<uint8_t>(m);
  s.dSBack = s.dalloc<uint8_t>(m);
  s.hPtsIn = s.halloc<Pt>(m);
  s.hFwd = s.halloc<Pt>(m);
  s.hBack = s.halloc<Pt>(m);
  s.hSFwd = s.halloc<uint8_t>(m);
  s.hSBack = s.halloc<uint8_t>(m);
  // Grid cells exactly as KltStereoTracker::detect.
  const double cw = static_cast<double>(s.W) / p.grid_cols, ch = static_cast<double>(s.H) / p.grid_rows;
  size_t maxPix = 0;
  for (int gy = 0; gy < p.grid_rows; ++gy)
    for (int gx = 0; gx < p.grid_cols; ++gx) {
      const CellRect r{static_cast<int>(gx * cw), static_cast<int>(gy * ch), static_cast<int>(cw), static_cast<int>(ch)};
      if (r.w < 3 || r.h < 3) throw std::invalid_argument("GpuKltCore: grid cells smaller than 3 px");
      s.rects.push_back(r);
      maxPix = std::max(maxPix, static_cast<size_t>(r.w) * r.h);
    }
  s.dRects = s.dalloc<CellRect>(cells);
  CUDA_CHECK(cudaMemcpy(s.dRects, s.rects.data(), cells * sizeof(CellRect), cudaMemcpyHostToDevice));
  s.cellStride = maxPix;
  s.dEig = s.dalloc<float>(maxPix * cells);
  s.dCellMax = s.dalloc<float>(cells);
  s.dDetPts = s.dalloc<Pt>(static_cast<size_t>(cells) * p.per_cell);
  s.dDetCount = s.dalloc<int>(cells);
  s.hDetPts = s.halloc<Pt>(static_cast<size_t>(cells) * p.per_cell);
  s.hDetCount = s.halloc<int>(cells);
  std::memset(s.hDetCount, 0, cells * sizeof(int));
  s.selectSmem = maxPix * sizeof(float);
  if (s.selectSmem > 48 * 1024)
    CUDA_CHECK(cudaFuncSetAttribute(k_select, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(s.selectSmem)));
  CUDA_CHECK(cudaEventRecord(s.uploadDone, s.stream));
}

GpuKltCore::~GpuKltCore() {
  Impl& s = *impl_;
  stopClockKeepAlive();
  cudaStreamSynchronize(s.stream);
  for (Impl::Pyr* py : {&s.P[0], &s.P[1], &s.PR, &s.PCL}) s.freePyr(*py);
  for (void* q : {static_cast<void*>(s.dRawL), static_cast<void*>(s.dRawR), static_cast<void*>(s.dMask),
                  static_cast<void*>(s.dNeed), static_cast<void*>(s.dSobel), static_cast<void*>(s.dKeys),
                  static_cast<void*>(s.dKeysSorted), static_cast<void*>(s.dAbsResp), static_cast<void*>(s.dS1),
                  static_cast<void*>(s.dS2), static_cast<void*>(s.dAcc), static_cast<void*>(s.dSigma),
                  static_cast<void*>(s.dFloor2), s.dSortTmp, static_cast<void*>(s.dHist), static_cast<void*>(s.dLut),
                  static_cast<void*>(s.dPtsIn), static_cast<void*>(s.dFwd), static_cast<void*>(s.dBack),
                  static_cast<void*>(s.dSFwd), static_cast<void*>(s.dSBack), static_cast<void*>(s.dRects),
                  static_cast<void*>(s.dEig), static_cast<void*>(s.dCellMax), static_cast<void*>(s.dDetPts),
                  static_cast<void*>(s.dDetCount)})
    cudaFree(q);
  for (void* q : {static_cast<void*>(s.hLeft), static_cast<void*>(s.hRight), static_cast<void*>(s.hMask),
                  static_cast<void*>(s.hNeed), static_cast<void*>(s.hSigma), static_cast<void*>(s.hPtsIn),
                  static_cast<void*>(s.hFwd), static_cast<void*>(s.hBack), static_cast<void*>(s.hSFwd),
                  static_cast<void*>(s.hSBack), static_cast<void*>(s.hDetPts), static_cast<void*>(s.hDetCount)})
    cudaFreeHost(q);
  for (auto& e : s.ev) cudaEventDestroy(e);
  cudaEventDestroy(s.uploadDone);
  cudaStreamDestroy(s.stream);
}

const CoreParams& GpuKltCore::params() const { return impl_->p; }
int GpuKltCore::numCells() const { return impl_->p.grid_cols * impl_->p.grid_rows; }
void GpuKltCore::cellRect(int cell, int* r) const {
  const CellRect& c = impl_->rects.at(static_cast<size_t>(cell));
  r[0] = c.x, r[1] = c.y, r[2] = c.w, r[3] = c.h;
}

uint8_t* GpuKltCore::leftBuffer() {
  impl_->waitUpload();
  return impl_->hLeft;
}
uint8_t* GpuKltCore::rightBuffer() {
  impl_->waitUpload();
  return impl_->hRight;
}
uint8_t* GpuKltCore::maskBuffer() {
  impl_->waitUpload();
  return impl_->hMask;
}
int* GpuKltCore::needBuffer() {
  impl_->waitUpload();
  return impl_->hNeed;
}

void GpuKltCore::upload() {
  Impl& s = *impl_;
  CUDA_CHECK(cudaMemcpyAsync(s.dRawL, s.hLeft, s.nPix, cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaMemcpyAsync(s.dRawR, s.hRight, s.nPix, cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaEventRecord(s.uploadDone, s.stream));
}

void GpuKltCore::frame(bool lcn, bool lcn_stereo, double lcn_min_floor) {
  Impl& s = *impl_;
  s.cur ^= 1;
  Impl::Pyr& tl = s.P[s.cur];
  s.stereoUsesClahe = lcn && !lcn_stereo;
  if (!lcn) {
    s.clahe(s.dRawL, tl.img[0]);
    s.clahe(s.dRawR, s.PR.img[0]);
  } else {
    s.noiseSigma(s.dRawL, lcn_min_floor);
    s.lcn(s.dRawL, tl.img[0]);
    if (lcn_stereo) {
      s.lcn(s.dRawR, s.PR.img[0]);
    } else {
      s.clahe(s.dRawL, s.PCL.img[0]);
      s.clahe(s.dRawR, s.PR.img[0]);
    }
  }
  s.buildLevels(tl);
  s.buildLevels(s.PR);
  if (s.stereoUsesClahe) s.buildLevels(s.PCL);
  s.sobel(tl.img[0]);
}

void GpuKltCore::reseedCurAsLcn(const uint8_t* raw, double lcn_min_floor) {
  Impl& s = *impl_;
  std::memcpy(leftBuffer(), raw, s.nPix);
  CUDA_CHECK(cudaMemcpyAsync(s.dRawL, s.hLeft, s.nPix, cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaEventRecord(s.uploadDone, s.stream));
  s.noiseSigma(s.dRawL, lcn_min_floor);
  Impl::Pyr& tl = s.P[s.cur];
  s.lcn(s.dRawL, tl.img[0]);
  s.buildLevels(tl);
}

void GpuKltCore::lkPair(PyrId from, PyrId to, int levels, const Pt* pts, int n) {
  Impl& s = *impl_;
  if (n > s.p.max_points) throw std::invalid_argument("GpuKltCore::lkPair: more points than max_points");
  if (n > 0) std::memcpy(s.hPtsIn, pts, static_cast<size_t>(n) * sizeof(Pt));
  s.lk(s.pyr(from), s.pyr(to), levels, n);
}

void GpuKltCore::trackStereo(const Pt* pts, int n) {
  Impl& s = *impl_;
  if (n > s.p.max_points) throw std::invalid_argument("GpuKltCore::trackStereo: more points than max_points");
  if (n > 0) std::memcpy(s.hPtsIn, pts, static_cast<size_t>(n) * sizeof(Pt));
  const Impl::Pyr& left = s.stereoUsesClahe ? s.PCL : s.P[s.cur];
  s.lk(left, s.PR, s.p.stereo_levels, n);
}

const Pt* GpuKltCore::fwd() const { return impl_->hFwd; }
const Pt* GpuKltCore::back() const { return impl_->hBack; }
const uint8_t* GpuKltCore::statusFwd() const { return impl_->hSFwd; }
const uint8_t* GpuKltCore::statusBack() const { return impl_->hSBack; }

void GpuKltCore::detect() {
  Impl& s = *impl_;
  const int cells = numCells();
  CUDA_CHECK(cudaMemcpyAsync(s.dMask, s.hMask, s.nPix, cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaMemcpyAsync(s.dNeed, s.hNeed, cells * sizeof(int), cudaMemcpyHostToDevice, s.stream));
  CUDA_CHECK(cudaEventRecord(s.uploadDone, s.stream));
  const float gscale = static_cast<float>(1.0 / (4.0 * 3.0 * 255.0));  // cornerMinEigenVal: 2^(ksize-1) * blockSize * 255
  k_eig<<<cells, 256, 0, s.stream>>>(s.dSobel, s.W, s.dMask, s.dRects, s.dNeed, s.dEig, s.cellStride, s.dCellMax, gscale);
  k_select<<<cells, 256, s.selectSmem, s.stream>>>(s.dEig, s.cellStride, s.dMask, s.W, s.dRects, s.dNeed, s.dCellMax,
                                                   s.p.quality_level, s.p.min_distance, s.p.per_cell, s.dDetPts,
                                                   s.dDetCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(s.hDetPts, s.dDetPts, static_cast<size_t>(cells) * s.p.per_cell * sizeof(Pt),
                             cudaMemcpyDeviceToHost, s.stream));
  CUDA_CHECK(cudaMemcpyAsync(s.hDetCount, s.dDetCount, cells * sizeof(int), cudaMemcpyDeviceToHost, s.stream));
}

const int* GpuKltCore::detCount() const { return impl_->hDetCount; }
const Pt* GpuKltCore::detPts() const { return impl_->hDetPts; }

void GpuKltCore::sync() { CUDA_CHECK(cudaStreamSynchronize(impl_->stream)); }
void GpuKltCore::startClockKeepAlive() {
  Impl& s = *impl_;
  if (s.spinFlag) return;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&s.spinFlag), sizeof(int), cudaHostAllocMapped));
  *s.spinFlag = 0;
  int* dflag = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dflag), s.spinFlag, 0));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.spinStream, cudaStreamNonBlocking));
  k_spin<<<1, 32, 0, s.spinStream>>>(dflag);
  CUDA_CHECK(cudaGetLastError());
}

void GpuKltCore::stopClockKeepAlive() {
  Impl& s = *impl_;
  if (!s.spinFlag) return;
  *const_cast<volatile int*>(s.spinFlag) = 1;
  cudaStreamSynchronize(s.spinStream);
  cudaStreamDestroy(s.spinStream);
  cudaFreeHost(s.spinFlag);
  s.spinStream = nullptr;
  s.spinFlag = nullptr;
}

void GpuKltCore::mark(Mark m) { CUDA_CHECK(cudaEventRecord(impl_->ev[static_cast<int>(m)], impl_->stream)); }
float GpuKltCore::elapsedMs(Mark a, Mark b) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->ev[static_cast<int>(a)], impl_->ev[static_cast<int>(b)]));
  return ms;
}
double GpuKltCore::lastSigma() const { return *impl_->hSigma; }

void GpuKltCore::loadPyramid(PyrId id, const uint8_t* img) {
  Impl& s = *impl_;
  Impl::Pyr& py = s.pyr(id);
  CUDA_CHECK(cudaMemcpyAsync(py.img[0], img, s.nPix, cudaMemcpyHostToDevice, s.stream));
  s.buildLevels(py);
  sync();
}
void GpuKltCore::levelSize(PyrId id, int level, int* w, int* h) const {
  const Impl::Pyr& py = impl_->pyr(id);
  if (level < 0 || level >= py.levels) throw std::out_of_range("GpuKltCore::levelSize");
  *w = py.w[level];
  *h = py.h[level];
}
int GpuKltCore::levelCount(PyrId id) const { return impl_->pyr(id).levels; }
void GpuKltCore::downloadLevel(PyrId id, int level, uint8_t* dst) {
  const Impl::Pyr& py = impl_->pyr(id);
  sync();
  CUDA_CHECK(cudaMemcpy(dst, py.img[level], static_cast<size_t>(py.w[level]) * py.h[level], cudaMemcpyDeviceToHost));
}
void GpuKltCore::downloadDeriv(PyrId id, int level, int16_t* dst) {
  const Impl::Pyr& py = impl_->pyr(id);
  sync();
  CUDA_CHECK(cudaMemcpy(dst, py.der[level], static_cast<size_t>(py.w[level]) * py.h[level] * sizeof(short2),
                        cudaMemcpyDeviceToHost));
}
void GpuKltCore::downloadEig(int cell, float* dst) {
  Impl& s = *impl_;
  sync();
  const CellRect& r = s.rects.at(static_cast<size_t>(cell));
  CUDA_CHECK(cudaMemcpy(dst, s.dEig + cell * s.cellStride, static_cast<size_t>(r.w) * r.h * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

}  // namespace gpu_frontend
