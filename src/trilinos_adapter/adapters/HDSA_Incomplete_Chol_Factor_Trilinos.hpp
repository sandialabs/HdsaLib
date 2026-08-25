/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_INCOMPLETE_CHOL_FACTOR_TRILINOS_HPP
#define HDSA_INCOMPLETE_CHOL_FACTOR_TRILINOS_HPP

#include "HDSA_Tpetra_Vector.hpp"
#include "Ifpack2_RILUK_decl.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"

namespace HDSA {

template <class RealT> class Incomplete_Chol_Factor_Trilinos : public HDSA::Incomplete_Chol_Factor<RealT> {

private:
  const HDSA::Ptr<Tpetra::CrsMatrix<RealT>> A_;
  HDSA::Ptr<Ifpack2::RILUK<Tpetra::RowMatrix<>>> riluk_prec;
  HDSA::Ptr<const Tpetra::CrsMatrix<RealT>> L_;
  HDSA::Ptr<Ifpack2::LocalSparseTriangularSolver<Tpetra::RowMatrix<>>> L_solver_;
  HDSA::Ptr<const Tpetra::CrsMatrix<RealT>> U_;
  HDSA::Ptr<Ifpack2::LocalSparseTriangularSolver<Tpetra::RowMatrix<>>> U_solver_;
  HDSA::Ptr<const Tpetra::Vector<RealT>> D_;

public:
  Incomplete_Chol_Factor_Trilinos(const HDSA::Ptr<Tpetra::CrsMatrix<RealT>>& A) : A_(A) {
    riluk_prec =
        HDSA::makePtr<Ifpack2::RILUK<Tpetra::RowMatrix<>>>(HDSA::dynamicPtrCast<const Tpetra::RowMatrix<>>(A_));
    riluk_prec->initialize();
    riluk_prec->compute();

    L_ = HDSA::makePtrFromRef(riluk_prec->getL());
    L_solver_ = HDSA::makePtr<Ifpack2::LocalSparseTriangularSolver<Tpetra::RowMatrix<>>>();
    L_solver_->setObjectLabel("lower");
    L_solver_->setMatrix(L_);
    L_solver_->initialize();
    L_solver_->compute();

    U_ = HDSA::makePtrFromRef(riluk_prec->getU());
    U_solver_ = HDSA::makePtr<Ifpack2::LocalSparseTriangularSolver<Tpetra::RowMatrix<>>>();
    U_solver_->setObjectLabel("upper");
    U_solver_->setMatrix(U_);
    U_solver_->initialize();
    U_solver_->compute();

    D_ = HDSA::makePtrFromRef(riluk_prec->getD());
  }

  virtual ~Incomplete_Chol_Factor_Trilinos() {}

  void Apply(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const {
    HDSA::Tpetra_Vector<RealT>& evec_out = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(vec_out);

    HDSA::Ptr<HDSA::Vector<RealT>> vec = vec_in.Clone();
    HDSA::Tpetra_Vector<RealT>& evec = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(*vec);

    // Scaling by square root of the diagonal
    vec->Set(vec_in);
    Teuchos::ArrayRCP<const RealT> vec_data = evec.getVector()->getData(0);
    Teuchos::ArrayRCP<const RealT> D_data = D_->getData();
    for (int i = 0; i < evec.getVector()->getLocalLength(); i++) {
      evec.getVector()->replaceLocalValue(i, 0, vec_data[i] / std::sqrt(D_data[i]));
    }

    // Applying L factor
    L_->apply(*evec.getVector(), *evec_out.getVector());
    // Picking up the diagonal contribution since L does not store the diagonal of ones
    vec_out.Plus(*vec);
  }

  void Apply_Inverse(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const {
    HDSA::Tpetra_Vector<RealT>& evec_out = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(vec_out);
    const HDSA::Tpetra_Vector<RealT>& evec_in = dynamic_cast<const HDSA::Tpetra_Vector<RealT>&>(vec_in);
    // Solving triangular system
    L_solver_->apply(*evec_in.getVector(), *evec_out.getVector());

    // Scaling by square root of the diagonal
    Teuchos::ArrayRCP<const RealT> vec_out_data = evec_out.getVector()->getData(0);
    Teuchos::ArrayRCP<const RealT> D_data = D_->getData();
    for (int i = 0; i < evec_out.getVector()->getLocalLength(); i++) {
      evec_out.getVector()->replaceLocalValue(i, 0, vec_out_data[i] * std::sqrt(D_data[i]));
    }
  }

  void Apply_Inverse_Transpose(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const {

    HDSA::Ptr<HDSA::Vector<RealT>> vec = vec_in.Clone();
    HDSA::Tpetra_Vector<RealT>& evec = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(*vec);

    // Scaling by square root of the diagonal
    vec->Set(vec_in);
    Teuchos::ArrayRCP<const RealT> vec_data = evec.getVector()->getData(0);
    Teuchos::ArrayRCP<const RealT> D_data = D_->getData();
    for (int i = 0; i < evec.getVector()->getLocalLength(); i++) {
      evec.getVector()->replaceLocalValue(i, 0, vec_data[i] * std::sqrt(D_data[i]));
    }

    // Solving triangular system
    HDSA::Tpetra_Vector<RealT>& evec_out = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(vec_out);
    U_solver_->apply(*evec.getVector(), *evec_out.getVector());
  }
};
} // namespace HDSA
#endif
