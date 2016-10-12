#pragma once
#include <functional>
#include <string>
#include <vector>
#include "types.hpp"

#include "cublas.cuh"
#include "fix1.cuh"
#include "fix2.cuh"
#include "fix_blend.cuh"
#include "fix_fb.cuh"
#include "fix_ip_ghost.cuh"
#include "var1.cuh"
#include "var_ip_ghost.cuh"
#include "varip1.cuh"

using MatmulFunctionType = std::function<bool(
    const size_t, const int, const int, const int, const dtype*, const int,
    const dtype, const dtype*, const int, const dtype, dtype*, const int)>;

std::vector<std::pair<MatmulFunctionType, std::string>> getEnabledVersions() {
  std::vector<std::pair<MatmulFunctionType, std::string>> versions;
#if PARM != 0 && PARN != 0
#ifdef FIX_BLEND
  versions.push_back({tsmm_fix_blend<dtype, PARM, PARN>, "FBLEND"});
#endif
#ifdef FIX_FB
  versions.push_back({tsmm_fix_fb<dtype, PARM, PARN>, "FIX_FB"});
#endif
#ifdef FIX1
  versions.push_back({tsmm_fix1<dtype, PARM, PARN>, "FIX_V1"});
#endif
#ifdef FIX2
  versions.push_back({tsmm_fix2<dtype, PARM, PARN>, "FIX_V2"});
#endif
#ifdef CUBLAS
  versions.push_back({tsmm_cublas<dtype>, "CUBLAS"});
#endif
#ifdef VAR1
  versions.push_back({tsmm_var1<dtype>, "VAR_V1"});
#endif
#ifdef FIXIPG
  versions.push_back({tsmm_fix_ip_ghost<dtype, PARM, PARN>, "FIXIPG"});
#endif
#ifdef VARIPG
  versions.push_back({tsmm_var_ip_ghost<dtype>, "VARIPG"});
#endif
#ifdef VARIP1
  versions.push_back({tsmm_varip1<dtype>, "VARIP1"});
#endif

#endif
  return versions;
}
