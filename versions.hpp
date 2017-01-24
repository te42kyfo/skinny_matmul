#pragma once
#include <functional>
#include <string>
#include <vector>
#include "types.hpp"
#ifdef TSMM
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
#endif
#ifdef TSMTTSM
#ifdef MAGMA
#include "tsmttsm/gen_magma.cuh"
#endif
// #include "tsmttsm/reduceonly.cuh"
#include "tsmttsm/gen_cublas.cuh"
#ifdef FIX_GENV1
#include "tsmttsm/genv1.cuh"
#endif
#include "tsmttsm/genv3.cuh"
#include "tsmttsm/genv32.cuh"
#include "tsmttsm/genv3x.cuh"
#include "tsmttsm/genv4.cuh"
#include "tsmttsm/genv5.cuh"
#include "tsmttsm/genv6.cuh"
#include "tsmttsm/specsmall.cuh"
#endif

using MatmulFunctionType = std::function<bool(
    const size_t, const int, const int, const int, const dtype*, const int,
    const dtype, const dtype*, const int, const dtype, dtype*, const int)>;

#ifdef TSMM
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
#endif

#ifdef TSMTTSM
std::vector<std::pair<MatmulFunctionType, std::string>>
getEnabledTSMTTSMVersions() {
  std::vector<std::pair<MatmulFunctionType, std::string>> versions;
#if PARM != 0 && PARN != 0
#ifdef CUBLAS
  versions.push_back({tsmttsm_cublas<dtype>, "CUBLAS"});
#endif
#ifdef MAGMA
  versions.push_back({tsmttsm_magma<dtype>, "MAGMA"});
#endif
#ifdef FIX_GENV1
  versions.push_back({GENV1::tsmttsm<dtype, PARM, PARN>, "FGENV1"});
#endif
#ifdef FIX_GENV3
  versions.push_back(
      {GENV3::tsmttsm<dtype, dItype, PARM, PARN, GENV3::MEMPATH::GLOBAL>,
       "FGENV3"});
#endif
#ifdef FIX_GENV3T
  versions.push_back(
      {GENV3::tsmttsm<dtype, dItype, PARM, PARN, GENV3::MEMPATH::TEX>,
       "FGENV3T"});
#endif
#ifdef FIX_GENV32T
  versions.push_back({GENV3X::tsmttsm<dtype, PARM, PARN, 2>, "FGENT32"});
#endif
#ifdef FIX_GENV4
  versions.push_back({GENV4::tsmttsm<dtype, PARM, PARN>, "FGENV4"});
#endif
#ifdef FIX_GENV5
  versions.push_back({GENV5::tsmttsm<dtype, PARM, PARN>, "FGENV5"});
#endif
#ifdef FIX_GENV6
  versions.push_back({GENV6::tsmttsm<dtype, PARM, PARN>, "FGENV6"});
#endif
#ifdef FIX_SPECSMALL
  versions.push_back({SPECSMALL::tsmttsm<dtype, dItype, PARM, PARN>, "FSMALL"});
#endif
//#ifdef FIX_REDUCEONLY
//  versions.push_back({REDUCEONLY::tsmttsm<dtype, dItype, PARM, PARN>,
//  "FREDUC"});
//#endif
#endif
  return versions;
}
#endif
