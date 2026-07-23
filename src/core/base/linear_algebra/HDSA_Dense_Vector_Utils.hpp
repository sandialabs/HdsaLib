/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_DENSE_VECTOR_UTILS_HPP
#define HDSA_DENSE_VECTOR_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "HDSA_Dense_Matrix.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Stack_Trace.hpp"

namespace HDSA {

template <class RealT> class Dense_Vector_Utils {
public:
  static int Length(const HDSA::Dense_Matrix<RealT>& x) { return x.Number_of_Rows() * x.Number_of_Columns(); }

  static RealT Get_Column_Major(const HDSA::Dense_Matrix<RealT>& x, const int idx) {
    return x(idx % x.Number_of_Rows(), idx / x.Number_of_Rows());
  }

  static void Set_Column_Major(HDSA::Dense_Matrix<RealT>& x, const int idx, const RealT val) {
    x.Set_Entry(idx % x.Number_of_Rows(), idx / x.Number_of_Rows(), val);
  }

  static void Check_Same_Length(const HDSA::Dense_Matrix<RealT>& x, const HDSA::Dense_Matrix<RealT>& y) {
    const int len_x = Length(x);
    const int len_y = Length(y);
    HDSA_TEST_FOR_EXCEPTION(len_x != len_y, std::logic_error,
                            "Error in HDSA::Dense_Vector_Utils::Check_Same_Length: "
                            "Dense vector lengths are incompatible."
                                << std::endl);
  }

  static RealT Dot(const HDSA::Dense_Matrix<RealT>& x, const HDSA::Dense_Matrix<RealT>& y) {
    Check_Same_Length(x, y);
    const int len = Length(x);
    RealT dot = static_cast<RealT>(0);
    for (int i = 0; i < len; ++i) {
      dot += Get_Column_Major(x, i) * Get_Column_Major(y, i);
    }
    return dot;
  }

  static RealT Norm(const HDSA::Dense_Matrix<RealT>& x) { return std::sqrt(Dot(x, x)); }

  static RealT Difference_Norm(const HDSA::Dense_Matrix<RealT>& x, const HDSA::Dense_Matrix<RealT>& y) {
    Check_Same_Length(x, y);
    const int len = Length(x);
    RealT norm_sq = static_cast<RealT>(0);
    for (int i = 0; i < len; ++i) {
      const RealT diff = Get_Column_Major(x, i) - Get_Column_Major(y, i);
      norm_sq += diff * diff;
    }
    return std::sqrt(norm_sq);
  }

  static RealT Max_Abs(const HDSA::Dense_Matrix<RealT>& x) {
    const int len = Length(x);
    RealT max_abs = static_cast<RealT>(0);
    for (int i = 0; i < len; ++i) {
      max_abs = std::max(max_abs, std::abs(Get_Column_Major(x, i)));
    }
    return max_abs;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Axpby(const RealT alpha, const HDSA::Dense_Matrix<RealT>& x,
                                                    const RealT beta, const HDSA::Dense_Matrix<RealT>& y) {
    Check_Same_Length(x, y);
    const int len = Length(x);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> z =
        HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(x.Number_of_Rows(), x.Number_of_Columns());
    for (int i = 0; i < len; ++i) {
      const RealT val = alpha * Get_Column_Major(x, i) + beta * Get_Column_Major(y, i);
      Set_Column_Major(*z, i, val);
    }
    return z;
  }

  static void Assign(HDSA::Dense_Matrix<RealT>& dst, const HDSA::Dense_Matrix<RealT>& src) {
    Check_Same_Length(dst, src);
    const int len = Length(src);
    for (int i = 0; i < len; ++i) {
      Set_Column_Major(dst, i, Get_Column_Major(src, i));
    }
  }

  static void Set_Scalar(HDSA::Dense_Matrix<RealT>& x, const RealT val) {
    const int len = Length(x);
    for (int i = 0; i < len; ++i) {
      Set_Column_Major(x, i, val);
    }
  }
};

} // namespace HDSA

#endif