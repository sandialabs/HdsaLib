/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_SPARSE_MATRIX_SOLVER_TRILINOS_HPP
#define HDSA_SPARSE_MATRIX_SOLVER_TRILINOS_HPP

#include "Amesos2_Factory.hpp"
#include "HDSA_Linear_Operator.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"

namespace HDSA {
template <class RealT> class Sparse_Matrix_Solver_Trilinos : public HDSA::Sparse_Matrix_Solver<RealT> {

private:
  HDSA::Ptr<Tpetra::CrsMatrix<RealT>> A_;
  HDSA::Ptr<Amesos2::Solver<Tpetra::CrsMatrix<>, Tpetra::MultiVector<>>> solver_;

public:
  Sparse_Matrix_Solver_Trilinos(const HDSA::Ptr<Tpetra::CrsMatrix<RealT>>& A, bool use_direct = true, int verbosity = 0,
                                std::ostream& out_stream = std::cout, const std::string solver_type_message = "")
      : HDSA::Sparse_Matrix_Solver<RealT>(verbosity, out_stream, solver_type_message) {
    A_ = A;
    HDSA::Sparse_Matrix_Solver<RealT>::Set_use_direct(use_direct);
    if (use_direct) {
      solver_ = Amesos2::create<Tpetra::CrsMatrix<>, Tpetra::MultiVector<>>("KLU2", A_);
      solver_->symbolicFactorization();
      solver_->numericFactorization();
    } else {
      HDSA::Ptr<HDSA::Linear_Operator<RealT>> A_op = HDSA::makePtr<Sparse_Matrix_Operator<RealT>>(this);
      HDSA::Sparse_Matrix_Solver<RealT>::Set_A_op(A_op);
    }
  }

  virtual ~Sparse_Matrix_Solver_Trilinos() {}

  void Sparse_Direct_Solve(HDSA::Vector<RealT>& x, const HDSA::Vector<RealT>& b) const override {
    HDSA::Tpetra_Vector<RealT>& ex = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(x);
    const HDSA::Tpetra_Vector<RealT>& eb = dynamic_cast<const HDSA::Tpetra_Vector<RealT>&>(b);
    solver_->setX(ex.getVector());
    solver_->setB(eb.getVector());
    solver_->solve();
  }

  template <class ScalarType> class Sparse_Matrix_Operator : public HDSA::Linear_Operator<ScalarType> {
  private:
    const Sparse_Matrix_Solver_Trilinos<ScalarType>* sparse_solver_;

  public:
    Sparse_Matrix_Operator(const Sparse_Matrix_Solver_Trilinos<ScalarType>* sparse_solver)
        : sparse_solver_(sparse_solver) {}

    ~Sparse_Matrix_Operator() {}

    void Apply(HDSA::Vector<ScalarType>& y, const HDSA::Vector<ScalarType>& x) const {
      const HDSA::Tpetra_Vector<RealT>& ex = dynamic_cast<const HDSA::Tpetra_Vector<RealT>&>(x);
      HDSA::Tpetra_Vector<RealT>& ey = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(y);
      sparse_solver_->A_->apply(*ex.getVector(), *ey.getVector());
    }
  };
};
} // namespace HDSA
#endif
