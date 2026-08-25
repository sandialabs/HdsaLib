/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OPT_PROB_INTERFACE_SYNTHETIC_TEST_HPP
#define HDSA_MD_OPT_PROB_INTERFACE_SYNTHETIC_TEST_HPP

#include "HDSA_MD_Opt_Prob_Interface.hpp"
#include "HDSA_Sparse_Matrix_Trilinos.hpp"

template <class RealT>
class MD_Opt_Prob_Interface_synthetic_test : public HDSA::MD_Opt_Prob_Interface<RealT>
{

private:
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_; // Mass matrix
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_sm_;
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_sm_;
  Teuchos::RCP<const Tpetra::Map<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> map_;

public:
  MD_Opt_Prob_Interface_synthetic_test(HDSA::Ptr<const HDSA::Comm<int>> &comm)
  {
    m_ = 51;
    RealT h = 1.0 / static_cast<RealT>(m_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }

    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    M_->Set_Entry(0, 0, (1.0 / 3.0) * h);
    M_->Set_Entry(0, 1, (1.0 / 6.0) * h);
    for (int i = 1; i < m_ - 1; i++)
    {
      M_->Set_Entry(i, i, (2.0 / 3.0) * h);
      M_->Set_Entry(i, i - 1, (1.0 / 6.0) * h);
      M_->Set_Entry(i, i + 1, (1.0 / 6.0) * h);
    }
    M_->Set_Entry(m_ - 1, m_ - 2, (1.0 / 6.0) * h);
    M_->Set_Entry(m_ - 1, m_ - 1, (1.0 / 3.0) * h);

    const int m = m_;
    map_ = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m, comm->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> M = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map_, 3); // 3 is the maximum number of non-zero entries per row
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> S = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map_, 3); // 3 is the maximum number of non-zero entries per row
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols0 = {0, 1};
    Teuchos::Array<RealT> vals0_M = {h / 3.0, h / 6.0};
    Teuchos::Array<RealT> vals0_S = {1.0 / h, -1.0 / h};
    if (M->getRowMap()->isNodeGlobalElement(0))
    {
      M->insertGlobalValues(0, cols0(), vals0_M());
      S->insertGlobalValues(0, cols0(), vals0_S());
    }
    for (int i = 1; i < m - 1; ++i)
    {
      Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols = {i - 1, i, i + 1};
      Teuchos::Array<RealT> vals_M = {h / 6.0, 2.0 * h / 3.0, h / 6.0};
      Teuchos::Array<RealT> vals_S = {-1.0 / h, 2.0 / h, -1.0 / h};
      if (M->getRowMap()->isNodeGlobalElement(i))
      {
        M->insertGlobalValues(i, cols(), vals_M());
        S->insertGlobalValues(i, cols(), vals_S());
      }
    }
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> colsm = {m - 2, m - 1};
    Teuchos::Array<RealT> valsm_M = {h / 6.0, h / 3.0};
    Teuchos::Array<RealT> valsm_S = {-1.0 / h, 1.0 / h};
    if (M->getRowMap()->isNodeGlobalElement(m - 1))
    {
      M->insertGlobalValues(m - 1, colsm(), valsm_M());
      S->insertGlobalValues(m - 1, colsm(), valsm_S());
    }
    M->fillComplete();
    S->fillComplete();

    M_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(M);
    S_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(S);
  }

  virtual ~MD_Opt_Prob_Interface_synthetic_test()
  {
  }

  // Assume a constraint u = z^3 nodewise on the mesh defined by nodes in x_
  // Assume an objective (1/2)*(u-T)^t*M*(u-T) where T = (x_+1.0)^3 so that the optimal solution is u_opt=(x_+1.0)^3 and z_opt=x_+1.0
  // Assume a high-fidelity model u = z^3 + .2*z^2

  void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Tpetra_Vector<RealT> z_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z);
    const HDSA::Tpetra_Vector<RealT> u_in_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u_in);
    HDSA::Tpetra_Vector<RealT> z_out_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z_out);
    Teuchos::ArrayRCP<const RealT> z_view = z_tpetra.getVector()->get1dView();
    Teuchos::ArrayRCP<const RealT> u_in_view = u_in_tpetra.getVector()->get1dView();
    for (int k = 0; k < m_; k++)
    {
      if (map_->isNodeGlobalElement(k))
      {
        Tpetra::Map<>::local_ordinal_type i = map_->getLocalElement(k);
        RealT val = 3.0 * std::pow(z_view[i], 2.0) * u_in_view[i];
        z_out_tpetra.getVector()->replaceGlobalValue(k, 0, val);
      }
    }
  }

  // This implementation assumes that it is evaluated at the optimal z so that the adjoint=0, a more general implementation would include a term multiplied by the adjoint variable
  void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Tpetra_Vector<RealT> z_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z);
    const HDSA::Tpetra_Vector<RealT> z_in_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z_in);
    HDSA::Tpetra_Vector<RealT> z_out_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z_out);
    HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_out.Clone();
    HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_out.Clone();
    HDSA::Tpetra_Vector<RealT> z_tmp1_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*z_tmp1);
    HDSA::Tpetra_Vector<RealT> z_tmp2_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*z_tmp2);
    Teuchos::ArrayRCP<const RealT> z_view = z_tpetra.getVector()->get1dView();
    Teuchos::ArrayRCP<const RealT> z_in_view = z_in_tpetra.getVector()->get1dView();
    Teuchos::ArrayRCP<const RealT> z_tmp2_view = z_tmp2_tpetra.getVector()->get1dView();

    for (int k = 0; k < m_; k++)
    {
      if (map_->isNodeGlobalElement(k))
      {
        Tpetra::Map<>::local_ordinal_type i = map_->getLocalElement(k);
        RealT val = 9.0 * (z_in_view[i] * std::pow(z_view[i], 2.0));
        z_tmp1_tpetra.getVector()->replaceGlobalValue(k, 0, val);
      }
    }
    M_sm_->Apply(*z_tmp2, *z_tmp1);
    for (int k = 0; k < m_; k++)
    {
      if (map_->isNodeGlobalElement(k))
      {
        Tpetra::Map<>::local_ordinal_type i = map_->getLocalElement(k);
        RealT val = z_tmp2_view[i] * std::pow(z_view[i], 2.0);
        z_out_tpetra.getVector()->replaceGlobalValue(k, 0, val);
      }
    }
  }

  void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Tpetra_Vector<RealT> u_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp1 = u_grad.Clone();
    HDSA::Tpetra_Vector<RealT> u_tmp1_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*u_tmp1);
    Teuchos::ArrayRCP<const RealT> u_view = u_tpetra.getVector()->get1dView();
    for (int k = 0; k < m_; k++)
    {
      if (map_->isNodeGlobalElement(k))
      {
        Tpetra::Map<>::local_ordinal_type i = map_->getLocalElement(k);
        RealT val = u_view[i] - std::pow((*x_)(k, 0) + 1.0, 3.0);
        u_tmp1_tpetra.getVector()->replaceGlobalValue(k, 0, val);
      }
    }
    M_sm_->Apply(u_grad, *u_tmp1);
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    M_sm_->Apply(u_out, u_in);
  }

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Get_Mass_Matrix(void) const
  {
    return M_sm_;
  }

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Get_Stiffness_Matrix(void) const
  {
    return S_sm_;
  }
};

#endif
