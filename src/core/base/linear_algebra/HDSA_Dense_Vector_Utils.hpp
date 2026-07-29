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

  static RealT Dot(const HDSA::Dense_Matrix<RealT>& x, const HDSA::Dense_Matrix<RealT>& y) {
    RealT dot = static_cast<RealT>(0);
    for (int i = 0; i < Length(x); ++i) {
      dot += Get_Column_Major(x, i) * Get_Column_Major(y, i);
    }
    return dot;
  }

  static RealT Norm(const HDSA::Dense_Matrix<RealT>& x) { return std::sqrt(Dot(x, x)); }

  static RealT Max_Abs(const HDSA::Dense_Matrix<RealT>& x) {
    RealT max_abs = static_cast<RealT>(0);
    for (int i = 0; i < Length(x); ++i) {
      max_abs = std::max(max_abs, std::abs(Get_Column_Major(x, i)));
    }
    return max_abs;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Concatenate(const HDSA::Dense_Matrix<RealT>& x,
                                                          const HDSA::Dense_Matrix<RealT>& y) {
    const int len_x = Length(x);
    const int len_y = Length(y);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> z = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(len_x + len_y, 1);
    for (int i = 0; i < len_x; ++i) {
      Set_Column_Major(*z, i, Get_Column_Major(x, i));
    }
    for (int i = 0; i < len_y; ++i) {
      Set_Column_Major(*z, len_x + i, Get_Column_Major(y, i));
    }
    return z;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Tail(const HDSA::Dense_Matrix<RealT>& x, const int tail_len) {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(tail_len, 1);
    const int start = Length(x) - tail_len;
    for (int i = 0; i < tail_len; ++i) {
      Set_Column_Major(*y, i, Get_Column_Major(x, start + i));
    }
    return y;
  }

  static RealT Column_Sum(const HDSA::Dense_Matrix<RealT>& A, const int col) {
    RealT val = static_cast<RealT>(0);
    for (int i = 0; i < A.Number_of_Rows(); ++i) {
      val += A(i, col);
    }
    return val;
  }

  static RealT Column_Dot(const HDSA::Dense_Matrix<RealT>& x, const HDSA::Dense_Matrix<RealT>& A, const int col) {
    RealT val = static_cast<RealT>(0);
    for (int i = 0; i < Length(x); ++i) {
      val += Get_Column_Major(x, i) * A(i, col);
    }
    return val;
  }

  static void Set_Columns_Column_Major(HDSA::Dense_Matrix<RealT>& A, const int first_col,
                                       const HDSA::Dense_Matrix<RealT>& x) {
    const int rows = A.Number_of_Rows();
    for (int idx = 0; idx < Length(x); ++idx) {
      A.Set_Entry(idx % rows, first_col + idx / rows, Get_Column_Major(x, idx));
    }
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Scale_Columns(const HDSA::Dense_Matrix<RealT>& A,
                                                            const HDSA::Dense_Matrix<RealT>& scales) {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> C =
        HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(A.Number_of_Rows(), A.Number_of_Columns());
    for (int j = 0; j < A.Number_of_Columns(); ++j) {
      const RealT scale = Get_Column_Major(scales, j);
      for (int i = 0; i < A.Number_of_Rows(); ++i) {
        C->Set_Entry(i, j, scale * A(i, j));
      }
    }
    return C;
  }
};

} // namespace HDSA

#endif