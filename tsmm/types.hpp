#pragma once
#include <string>
#include "../cu_complex.h"

#ifdef FC
typedef complex<float> htype;
typedef cuFloatComplex dtype;
dtype makeDtype(htype v) { return make_cuFloatComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuFloatComplex(v1, v2)
std::string mode = "float complex";
int flopsPerCell = 8;

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)
std::string mode = "double complex";
int flopsPerCell = 8;

#elif FR
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)
std::string mode = "float real";
int flopsPerCell = 2;

#elif DR
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
std::string mode = "double real";
int flopsPerCell = 2;

#endif
