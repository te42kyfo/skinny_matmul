#pragma once

#include "cu_complex.h"

struct PseudoQuad {
  double s;
  double t;
};

__device__ PseudoQuad Fast2Sum(double a, double b) {
  double s = __dadd_rn(a, b);
  double z = __dadd_rn(s, -a);
  double t = __dadd_rn(b, -z);
  return {s, t};
}

__device__ PseudoQuad Gen2Sum(double a, double b) {
  double s = __dadd_rn(a, b);
  double ap = __dadd_rn(s, -b);
  double bp = __dadd_rn(s, -ap);
  double da = __dadd_rn(a, -ap);
  double db = __dadd_rn(b, -bp);
  double t = __dadd_rn(da, db);
  return {s, t};
}

__device__ PseudoQuad FMA2Mult(double a, double b) {
  double s = __dmul_rn(a, b);
  double t = __fma_rn(a, b, -s);
  return {s, t};
}

// --- ZERO - return zero
// ----------------------

template <>
__device__ __host__ inline void zero<PseudoQuad>(PseudoQuad &val) {
  val.s = 0.0;
  val.t = 0.0;
}

// --- fromReal - return exactly this value
// ----------------------------------------
template <>
__device__ inline void fromReal<PseudoQuad, double>(PseudoQuad &val,
                                                    double real) {
  val.s = real;
  val.t = 0;
}

// --- ACCU --- T S = A + B,  Melvens version
// ------------------------------------------
// template <>
__device__ PseudoQuad accu_fast(PseudoQuad X, PseudoQuad Y) {
  PseudoQuad S = Fast2Sum(X.s, Y.s);
  PseudoQuad T = Fast2Sum(X.t, Y.t);
  PseudoQuad V = Fast2Sum(S.t, T.s);
  double W = __dadd_rn(T.t, V.t);
  PseudoQuad R = Fast2Sum(S.s, V.s);
  R.t = __dadd_rn(R.t, W);
  return R;
}

// --- ACCU --- T S = A + B,  Book version
// ------------------------------------------
template <>
__device__ PseudoQuad accu(PseudoQuad X, PseudoQuad Y) {
  PseudoQuad S = Gen2Sum(X.s, Y.s);
  PseudoQuad T = Gen2Sum(X.t, Y.t);
  double C = __dadd_rn(S.t, T.s);
  PseudoQuad V = Fast2Sum(S.s, C);
  double W = __dadd_rn(T.t, V.t);
  PseudoQuad R = Fast2Sum(V.s, W);
  return R;
}

// --- SCALE - T S = A * B
// -----------------------

template <>
__device__ PseudoQuad scale2(PseudoQuad A, double b) {
  PseudoQuad p = FMA2Mult(A.s, b);
  p.t = __fma_rn(A.s, 0, p.t);
  p.t = __fma_rn(A.t, b, p.t);
  return Fast2Sum(p.s, p.t);
};

// --- AXPY -  T S = a*b + c
// -------------------------

template <typename T1, typename T2>
__device__ inline T1 axpy2(T1 val, T2 val2, T2 val3) {
  return val + val2 * val3;
}

template <>
__device__ PseudoQuad axpy2(PseudoQuad C, double A, double B) {
  PseudoQuad R;
  zero(R);
  R.s = __fma_rn(A, B, C.s);
  double ABP = __dadd_rn(R.s, -C.s);
  double CP = __dadd_rn(R.s, -ABP);
  double DAB = __fma_rn(A, B, -ABP);
  double DC = __dadd_rn(C.s, -CP);
  R.t = __dadd_rn(DC, DAB);
  R.t = __dadd_rn(R.t, C.t);
  return R;
}

template <>
__device__ PseudoQuad axpy(PseudoQuad C, PseudoQuad A, double B) {
  PseudoQuad XIJ = FMA2Mult(A.s, B);
  XIJ.t = __fma_rn(B, A.t, XIJ.t);
  PseudoQuad result = Fast2Sum(C.s, XIJ.s);
  result.t = __dadd_rn(result.t, XIJ.t);
  result.t = __dadd_rn(result.t, C.t);
  return result;
}

// --- RENORMALIZE - renormalizes s and t
// --------------------------------------
template <typename T>
__device__ T renormalize(T a) {
  return a;
}

template <>
__device__ PseudoQuad renormalize(PseudoQuad a) {
  return Fast2Sum(a.s, a.t);
}

// --- CONVERT - extends/truncates to target format
// ------------------------------------------------
template <typename iT, typename oT>
__device__ oT convert(iT a) {
  return a;
}

template <>
__device__ double convert(PseudoQuad a) {
  return __dadd_rn(a.s, a.t);
}

template <>
__device__ PseudoQuad convert(double a) {
  return {a, 0.0};
}
