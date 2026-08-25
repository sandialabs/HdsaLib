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
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  std::vector<RealT> t_;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_; // Mass matrix
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_sm_;
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_sm_;
  std::vector<std::vector<RealT>> H_;

public:
  MD_Opt_Prob_Interface_synthetic_test(HDSA::Ptr<const HDSA::Comm<int>> &comm, HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, int n_y, int n_t) : data_interface_(data_interface)
  {
    n_y_ = n_y;
    n_t_ = n_t;
    RealT h = 1.0 / static_cast<RealT>(n_y_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, 1);
    for (int k = 0; k < n_y_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(n_y_ - 1));
    }
    t_ = std::vector<RealT>(n_t_);
    for (int k = 0; k < n_t_; k++)
    {
      t_[k] = static_cast<RealT>(k) / static_cast<RealT>(n_t_ - 1);
    }

    H_.resize(2);
    H_[0].resize(2);
    H_[1].resize(2);
    RealT val1 = 0.0;
    RealT val2 = 0.0;
    for (int k = 0; k < n_y_; k++)
    {
      val1 += (*x_)(k, 0) * (*x_)(k, 0);
      val2 += (*x_)(k, 0) * (*x_)(n_y_ - k - 1, 0);
    }
    H_[0][0] = val1;
    H_[1][1] = val1;
    H_[0][1] = val2;
    H_[1][0] = val2;

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
    HDSA::Transient_Vector<RealT> z_out_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(z_out);
    const HDSA::Transient_Vector<RealT> u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u_in);

    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> uj = u_in_trans[j];
      HDSA::Ptr<HDSA::Vector<RealT>> zj = z_out_trans[j];
      HDSA::Tpetra_Vector<RealT> uj_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*uj);
      Teuchos::ArrayRCP<const RealT> u_in_view = uj_tpetra.getVector()->get1dView();
      RealT val0 = 0.0;
      RealT val1 = 0.0;
      for (int k = 0; k < n_y_; k++)
      {
        val0 += u_in_view[k] * (1.0 - (*x_)(k, 0));
        val1 += u_in_view[k] * (*x_)(k, 0);
      }
      zj->Set_Entry(0, val0);
      zj->Set_Entry(1, val1);
    }
  }

  void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
  {
    HDSA::Transient_Vector<RealT> z_out_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(z_out);
    const HDSA::Transient_Vector<RealT> z_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(z_in);

    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_outj = z_out_trans[j];
      HDSA::Ptr<HDSA::Vector<RealT>> z_inj = z_in_trans[j];
      RealT val0 = H_[0][0] * z_inj->Get_Entry(0) + H_[0][1] * z_inj->Get_Entry(1);
      RealT val1 = H_[1][0] * z_inj->Get_Entry(0) + H_[1][1] * z_inj->Get_Entry(1);
      z_outj->Set_Entry(0, val0);
      z_outj->Set_Entry(1, val1);
    }
  }

  void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    u_grad.Set(u);
    u_grad.Scaled_Plus(-1.0, *data_interface_->Get_u_opt());
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    u_out.Set(u_in);
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
