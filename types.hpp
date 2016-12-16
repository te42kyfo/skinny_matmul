#pragma once
#include <string>
#include "cu_complex.h"
#include "PseudoQuad.cuh"

#define XSTR(s) STR(s)
#define STR(s) #s

#ifdef FC
typedef complex<float> htype;
typedef cuFloatComplex dtype;
typedef cuFloatComplex dItype;
dtype makeDtype(htype v) { return make_cuFloatComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuFloatComplex(v1, v2)
std::string types = "FC";
int flopsPerCell = 8;

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
typedef cuDoubleComplex dItype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)
std::string types = "DC";
int flopsPerCell = 8;

#elif FR
typedef float htype;
typedef float dtype;
typedef float dItype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)
std::string types = "FR";
int flopsPerCell = 2;

#elif DR
typedef double htype;
typedef double dtype;
typedef double dItype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
std::string types = "DR";
int flopsPerCell = 2;

#elif HR
typedef double htype;
typedef double dtype;
typedef PseudoQuad dItype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
std::string types = "HR";
int flopsPerCell = 9;

#endif
