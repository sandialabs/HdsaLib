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
  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
  int n_y_; // Mesh resolution
  int n_t_;
  RealT c_low_;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_; // Mass matrix
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_sm_;
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_sm_;

public:
  MD_Opt_Prob_Interface_synthetic_test(HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, HDSA::Ptr<const HDSA::Comm<int>> &comm, int n_y, int n_t, RealT c_low) : data_interface_(data_interface), n_y_(n_y), n_t_(n_t), c_low_(c_low)
  {
    RealT h = 1.0 / static_cast<RealT>(n_y_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, 1);
    for (int k = 0; k < n_y_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(n_y_ - 1));
    }

    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, n_y_);
    M_->Set_Entry(0, 0, (1.0 / 3.0) * h);
    M_->Set_Entry(0, 1, (1.0 / 6.0) * h);
    for (int i = 1; i < n_y_ - 1; i++)
    {
      M_->Set_Entry(i, i, (2.0 / 3.0) * h);
      M_->Set_Entry(i, i - 1, (1.0 / 6.0) * h);
      M_->Set_Entry(i, i + 1, (1.0 / 6.0) * h);
    }
    M_->Set_Entry(n_y_ - 1, n_y_ - 2, (1.0 / 6.0) * h);
    M_->Set_Entry(n_y_ - 1, n_y_ - 1, (1.0 / 3.0) * h);

    const int m = n_y_;
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m, comm->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> M = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map, 3); // 3 is the maximum number of non-zero entries per row
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> S = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map, 3); // 3 is the maximum number of non-zero entries per row
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols0 = {0, 1};
    Teuchos::Array<RealT> vals0_M = {h / 3.0, h / 6.0};
    M->insertGlobalValues(0, cols0(), vals0_M());
    Teuchos::Array<RealT> vals0_S = {1.0 / h, -1.0 / h};
    S->insertGlobalValues(0, cols0(), vals0_S());
    for (int i = 1; i < m - 1; ++i)
    {
      Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols = {i - 1, i, i + 1};
      Teuchos::Array<RealT> vals_M = {h / 6.0, 2.0 * h / 3.0, h / 6.0};
      M->insertGlobalValues(i, cols(), vals_M());
      Teuchos::Array<RealT> vals_S = {-1.0 / h, 2.0 / h, -1.0 / h};
      S->insertGlobalValues(i, cols(), vals_S());
    }
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> colsm = {m - 2, m - 1};
    Teuchos::Array<RealT> valsm_M = {h / 6.0, h / 3.0};
    M->insertGlobalValues(m - 1, colsm(), valsm_M());
    Teuchos::Array<RealT> valsm_S = {-1.0 / h, 1.0 / h};
    S->insertGlobalValues(m - 1, colsm(), valsm_S());
    M->fillComplete();
    S->fillComplete();

    M_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(M);
    S_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(S);
  }

  virtual ~MD_Opt_Prob_Interface_synthetic_test()
  {
  }

  void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Tpetra_Vector<RealT> z_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z);
    const HDSA::Transient_Vector<RealT> u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u_in);
    HDSA::Tpetra_Vector<RealT> z_out_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(z_out);
    Teuchos::ArrayRCP<const RealT> z_view = z_tpetra.getVector()->get1dView();
    RealT coeff = 1.0;
    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> uj = u_in_trans[j];
      HDSA::Tpetra_Vector<RealT> uj_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*uj);
      Teuchos::ArrayRCP<const RealT> u_in_view = uj_tpetra.getVector()->get1dView();
      for (int k = 0; k < n_y_; k++)
      {
        RealT val = coeff * 3.0 * std::pow(z_view[k], 2.0) * u_in_view[k];
        z_out_tpetra.getVector()->replaceGlobalValue(k, 0, val);
        val = c_low_ * coeff * 3.0 * std::pow(z_view[k], 2.0) * u_in_view[n_y_ + k];
        z_out_tpetra.getVector()->replaceGlobalValue(k, 0, val);
      }
      coeff = c_low_ * coeff;
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

    for (int k = 0; k < n_y_; k++)
    {
      RealT val = std::pow(c_low_, n_t_) * 3.0 * (z_in_view[k] * std::pow(z_view[k], 2.0));
      z_tmp1_tpetra.getVector()->replaceGlobalValue(k, 0, val);
    }
    M_sm_->Apply(*z_tmp2, *z_tmp1);
    for (int k = 0; k < n_y_; k++)
    {
      RealT val = std::pow(c_low_, n_t_) * 3.0 * z_tmp2_view[k] * std::pow(z_view[k], 2.0);
      z_out_tpetra.getVector()->replaceGlobalValue(k, 0, val);
    }
  }

  void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Transient_Vector<RealT> u_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u);
    HDSA::Ptr<const HDSA::Vector<RealT>> uf = data_interface_->Extract_State_Component(*u_trans[n_t_ - 1], 1);
    const HDSA::Tpetra_Vector<RealT> uf_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*uf);
    Teuchos::ArrayRCP<const RealT> uf_view = uf_tpetra.getVector()->get1dView();

    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp1 = uf->Clone();
    HDSA::Tpetra_Vector<RealT> u_tmp1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*u_tmp1);
    for (int k = 0; k < n_y_; k++)
    {
      RealT val = uf_view[k] - std::pow(c_low_, n_t_) * std::pow((*x_)(k, 0) + 1.0, 3.0);
      u_tmp1_tpetra.getVector()->replaceGlobalValue(k, 0, val);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = uf->Clone();
    M_sm_->Apply(*u_tmp2, *u_tmp1);
    const HDSA::Transient_Vector<RealT> u_grad_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u_grad);
    HDSA::Ptr<HDSA::Vector<RealT>> uf_grad = u_grad_trans[n_t_ - 1];
    data_interface_->Set_State_Component(*uf_grad, *u_tmp2, 1);
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    const HDSA::Transient_Vector<RealT> u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u_in);
    HDSA::Ptr<HDSA::Vector<RealT>> uf_in = u_in_trans[n_t_ - 1];
    HDSA::Ptr<const HDSA::Vector<RealT>> u_in2 = data_interface_->Extract_State_Component(*uf_in, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = u_in2->Clone();
    M_sm_->Apply(*u_tmp, *u_in2);
    const HDSA::Transient_Vector<RealT> u_out_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u_out);
    HDSA::Ptr<HDSA::Vector<RealT>> uf_out = u_out_trans[n_t_ - 1];
    data_interface_->Set_State_Component(*uf_out, *u_tmp, 1);
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
