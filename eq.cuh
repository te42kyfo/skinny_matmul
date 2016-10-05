#pragma once
#include "cu_complex.h"

template <typename T>
bool eq(const T lhs, const T rhs) {
  return lhs == rhs;
};

template <>
bool eq<cuDoubleComplex>(const cuDoubleComplex lhs, const cuDoubleComplex rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <>
bool eq<cuFloatComplex>(const cuFloatComplex lhs, const cuFloatComplex rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}
