/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_SPARSE_MATRIX_HPP
#define HDSA_SPARSE_MATRIX_HPP

#include "HDSA_Incomplete_Chol_Factor.hpp"
#include "HDSA_Sparse_Matrix_Solver.hpp"
#include "HDSA_Vector.hpp"

namespace HDSA {

template <class RealT> class Sparse_Matrix {

private:
  bool is_symmetric_;

public:
  Sparse_Matrix(void) {}

  virtual ~Sparse_Matrix() {}

  void Set_Symmetric(void) { is_symmetric_ = true; }

  bool Is_Symmetric(void) const { return is_symmetric_; }

  virtual HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Clone(int max_entries_per_row = 0) const = 0;

  // Compute C = this * B, with options for transposes
  virtual void Matrix_Matrix_Multiply(HDSA::Sparse_Matrix<RealT>& C, const HDSA::Sparse_Matrix<RealT>& B,
                                      bool A_trans = false, bool B_trans = false) const = 0;

  // Compute x_out = this * x_in
  virtual void Apply(HDSA::Vector<RealT>& x_out, const HDSA::Vector<RealT>& x_in) const = 0;

  virtual void Set(HDSA::Sparse_Matrix<RealT>& B) = 0;

  virtual void Set_Diagonal(HDSA::Vector<RealT>& vec, bool reciprocate_diag) = 0;

  virtual void Scaled_Plus(const RealT& alpha, const HDSA::Sparse_Matrix<RealT>& B) = 0;

  virtual int Get_Max_Nonzeros_Per_Row(void) const = 0;

  virtual HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> Get_Incomplete_Chol_Factor(void) const = 0;

  virtual HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>>
  Get_Sparse_Matrix_Solver(bool use_direct = true, int verbosity = 0, std::ostream& out_stream = std::cout,
                           const std::string solver_type_message = "") const = 0;
};

} // namespace HDSA

#endif
