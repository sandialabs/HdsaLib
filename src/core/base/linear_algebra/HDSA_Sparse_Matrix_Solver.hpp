/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_SPARSE_MATRIX_SOLVER_HPP
#define HDSA_SPARSE_MATRIX_SOLVER_HPP

#include "HDSA_Incomplete_Chol_Factor.hpp"
#include "HDSA_Linear_Algebra.hpp"

namespace HDSA {
template <class RealT> class Sparse_Matrix_Solver {

private:
  HDSA::Ptr<HDSA::Linear_Operator<RealT>> A_op_;
  bool use_direct_;
  HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L_;
  bool use_incomplete_factorization_;
  int verbosity_;
  std::ostream& out_stream_;
  const std::string solver_type_message_;

public:
  Sparse_Matrix_Solver(int verbosity = 0, std::ostream& out_stream = std::cout,
                       const std::string solver_type_message = "")
      : verbosity_(verbosity), out_stream_(out_stream), solver_type_message_(solver_type_message) {
    use_incomplete_factorization_ = false;
  }

  Sparse_Matrix_Solver(const HDSA::Ptr<HDSA::Linear_Operator<RealT>>& A_op, int verbosity = 0,
                       std::ostream& out_stream = std::cout, const std::string solver_type_message = "")
      : A_op_(A_op), use_direct_(false), verbosity_(verbosity), out_stream_(out_stream),
        solver_type_message_(solver_type_message) {
    use_incomplete_factorization_ = false;
  }

  virtual ~Sparse_Matrix_Solver() {}

  virtual void Sparse_Direct_Solve(HDSA::Vector<RealT>& x, const HDSA::Vector<RealT>& b) const {
    HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                            "Error in HDSA::Sparse_Matrix_Solver: A sparse direct solve was requested, but the "
                            "Sparse_Direct_Solve method has not been implemented"
                                << std::endl);
    (void) x;
    (void) b;
  }

  void Set_A_op(const HDSA::Ptr<HDSA::Linear_Operator<RealT>> &A_op)
  {
    A_op_ = A_op;
  }

  void Set_use_direct(bool use_direct)
  {
    use_direct_ = use_direct;
  }

  void Set_Incomplete_Factor(HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>>& L) {
    if (use_direct_) {
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::Sparse_Matrix_Solver: An incomplete factorization was set but the solver "
                              "is configured for sparse direct solves"
                                  << std::endl);
    }
    L_ = L;
    use_incomplete_factorization_ = true;
  }

  bool Use_Incomplete_Factor(void) const { return use_incomplete_factorization_; }

  std::string Apply_A_Inverse(HDSA::Vector<RealT>& x, const HDSA::Vector<RealT>& b) {
    std::string output_message;
    std::string output_message_solver;
    if (use_direct_) {
      Sparse_Direct_Solve(x, b);
    } else {
      RealT tol = 1.0E-10;
      std::string solver = "GMRES";
      if (A_op_->Is_Symmetric()) { solver = "CG"; }
      HDSA::Ptr<A_Operator<RealT>> A_op = HDSA::makePtr<A_Operator<RealT>>(this);
      if (use_incomplete_factorization_) {
        HDSA::Ptr<HDSA::Vector<RealT>> b_prec = b.Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> x_prec = x.Clone();
        L_->Apply_Inverse(*b_prec, b);
        output_message_solver = HDSA::Linear_Algebra::Iterative_Linear_Solve<RealT>(*x_prec, *b_prec, *A_op, tol,
                                                                                    solver, verbosity_, out_stream_);
        L_->Apply_Inverse_Transpose(x, *x_prec);
      } else {
        output_message_solver =
            HDSA::Linear_Algebra::Iterative_Linear_Solve<RealT>(x, b, *A_op, tol, solver, verbosity_, out_stream_);
      }

      output_message = solver_type_message_ + "::" + output_message_solver;
    }
    return output_message;
  }

  template <class ScalarType> class A_Operator : public HDSA::Linear_Operator<ScalarType> {
  private:
    const Sparse_Matrix_Solver<ScalarType>* A_invert_;

  public:
    A_Operator(const Sparse_Matrix_Solver<ScalarType>* A_invert) : A_invert_(A_invert) {}

    ~A_Operator() {}

    void Apply(HDSA::Vector<ScalarType>& y, const HDSA::Vector<ScalarType>& x) const {

      if (A_invert_->Use_Incomplete_Factor()) {
        HDSA::Ptr<HDSA::Vector<RealT>> vec_tmp1 = y.Clone();
        A_invert_->L_->Apply_Inverse_Transpose(*vec_tmp1, x);
        HDSA::Ptr<HDSA::Vector<RealT>> vec_tmp2 = y.Clone();
        A_invert_->A_op_->Apply(*vec_tmp2, *vec_tmp1);
        A_invert_->L_->Apply_Inverse(y, *vec_tmp2);
      } else {
        A_invert_->A_op_->Apply(y, x);
      }
    }
  };
};
} // namespace HDSA
#endif
