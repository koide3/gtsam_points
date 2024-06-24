// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/cuda/check_error_cusolver.cuh>

#include <cusparse.h>

namespace gtsam_points {

std::string cusolverGetErrorName(int error) {
  switch (error) {
    default:
      return "CUSOLVER_UNKNOWN_ERROR";
    case 0:
      return "CUSOLVER_STATUS_SUCCESS";
    case 1:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case 2:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case 3:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case 4:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case 5:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case 6:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case 7:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case 8:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case 9:
      return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case 10:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case 11:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
    case 12:
      return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
    case 13:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
    case 14:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
    case 15:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
    case 16:
      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
    case 20:
      return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
    case 21:
      return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
    case 22:
      return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
    case 23:
      return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
    case 25:
      return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
    case 26:
      return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
    case 30:
      return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
    case 31:
      return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  }
}

void CusolverCheckError::operator<<(int error) const {
  if (error == 0) {
    return;
  }

  const std::string error_name = cusolverGetErrorName(error);
  std::cerr << "warning: " << error_name << std::endl;
}

CusolverCheckError check_cusolver;

}  // namespace gtsam_points