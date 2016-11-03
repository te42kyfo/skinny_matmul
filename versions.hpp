#pragma once
#include <functional>
#include <string>
#include <vector>
#include "types.hpp"

#include "tsmm/cublas.cuh"
#include "tsmm/fix1.cuh"
#include "tsmm/fix2.cuh"
#include "tsmm/fix_blend.cuh"
#include "tsmm/fix_fb.cuh"
#include "tsmm/fix_ip_ghost.cuh"
#include "tsmm/var1.cuh"
#include "tsmm/var_ip_ghost.cuh"
#include "tsmm/varip1.cuh"
#include "tsmm/varip2.cuh"
#include "tsmm/varip3.cuh"
#include "tsmm/varip_blend.cuh"
#include "tsmttsm/gen_cublas.cuh"
#include "tsmttsm/genv3.cuh"

using MatmulFunctionType = std::function<bool(
    const size_t, const int, const int, const int, const dtype*, const int,
    const dtype, const dtype*, const int, const dtype, dtype*, const int)>;

std::vector<std::pair<MatmulFunctionType, std::string>>
getEnabledTSMMVersions() {
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
#ifdef VARIP2
  versions.push_back({tsmm_varip2<dtype>, "VARIP2"});
#endif
#ifdef VARIP3
  versions.push_back({tsmm_varip3<dtype>, "VARIP3"});
#endif
#ifdef VARIP_BLEND
  versions.push_back({tsmm_varip_blend<dtype>, "VARIPB"});
#endif

#endif
  return versions;
}

std::vector<std::pair<MatmulFunctionType, std::string>>
getEnabledTSMTTSMVersions() {
  std::vector<std::pair<MatmulFunctionType, std::string>> versions;
#if PARM != 0 && PARN != 0
#ifdef CUBLAS
  versions.push_back({tsmttsm_cublas<dtype>, "CUBLAS"});
#endif
#ifdef FIX_GENV3
  versions.push_back({GENV3::tsmttsm<dtype, PARM, PARN>, "FGENV3"});
#endif
#endif
  return versions;
}
