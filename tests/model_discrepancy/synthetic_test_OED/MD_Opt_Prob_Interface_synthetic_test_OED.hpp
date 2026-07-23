/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OPT_PROB_INTERFACE_SYNTHETIC_TEST_OED_HPP
#define HDSA_MD_OPT_PROB_INTERFACE_SYNTHETIC_TEST_OED_HPP

#include "HDSA_MD_Opt_Prob_Interface.hpp"

template <class RealT> class MD_Opt_Prob_Interface_synthetic_test_OED : public HDSA::MD_Opt_Prob_Interface<RealT> {

private:
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_; // Mass matrix

public:
  MD_Opt_Prob_Interface_synthetic_test_OED() {
    m_ = 51;
    RealT h = 1.0 / static_cast<RealT>(m_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++) {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }

    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    M_->Set_Entry(0, 0, (1.0 / 3.0) * h);
    M_->Set_Entry(0, 1, (1.0 / 6.0) * h);
    for (int i = 1; i < m_ - 1; i++) {
      M_->Set_Entry(i, i, (2.0 / 3.0) * h);
      M_->Set_Entry(i, i - 1, (1.0 / 6.0) * h);
      M_->Set_Entry(i, i + 1, (1.0 / 6.0) * h);
    }
    M_->Set_Entry(m_ - 1, m_ - 2, (1.0 / 6.0) * h);
    M_->Set_Entry(m_ - 1, m_ - 1, (1.0 / 3.0) * h);
  }

  virtual ~MD_Opt_Prob_Interface_synthetic_test_OED() {}

  // Assume a constraint u = z^3 nodewise on the mesh defined by nodes in x_
  // Assume an objective (1/2)*(u-T)^t*M*(u-T) where T = (x_+1.0)^3 so that the optimal solution is u_opt=(x_+1.0)^3 and
  // z_opt=x_+1.0 Assume a high-fidelity model u = z^3 + .2*z^3 (NOTE: not u = z^3 + .2*z^2)

  void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT>& z_out, const HDSA::Vector<RealT>& u_in,
                                                    const HDSA::Vector<RealT>& z) const {
    const HDSA::Std_Vector<RealT> u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(u_in);
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z);
    HDSA::Std_Vector<RealT> z_out_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(z_out);
    for (int k = 0; k < m_; k++) {
      z_out_std.Set_Entry(k, 3.0 * std::pow(z_std(k), 2.0) * u_in_std(k));
    }
  }

  void Apply_Solution_Operator_z_Jacobian(HDSA::Vector<RealT>& u_out, const HDSA::Vector<RealT>& z_in,
                                          const HDSA::Vector<RealT>& z) const {
    const HDSA::Std_Vector<RealT> z_in_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z_in);
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z);
    HDSA::Std_Vector<RealT> u_out_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(u_out);
    for (int k = 0; k < m_; k++) {
      u_out_std.Set_Entry(k, 3.0 * std::pow(z_std(k), 2.0) * z_in_std(k));
    }
  }

  // This implementation assumes that it is evaluated at the optimal z so that the adjoint=0, a more general
  // implementation would include a term multiplied by the adjoint variable
  void Apply_RS_Hessian(HDSA::Vector<RealT>& z_out, const HDSA::Vector<RealT>& z_in,
                        const HDSA::Vector<RealT>& z) const {
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z);
    const HDSA::Std_Vector<RealT> z_in_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z_in);
    HDSA::Std_Vector<RealT> z_out_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(z_out);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++) {
      v->Set_Entry(k, 0, 9.0 * (z_in_std(k) * std::pow(z_std(k), 2.0)));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*M_v, *v);
    for (int k = 0; k < m_; k++) {
      z_out_std.Set_Entry(k, (*M_v)(k, 0) * std::pow(z_std(k), 2.0));
    }
  }

  void Misfit_Gradient(HDSA::Vector<RealT>& u_grad, const HDSA::Vector<RealT>& u, const HDSA::Vector<RealT>& z) const {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> u_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(u);
    HDSA::Std_Vector<RealT> u_grad_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(u_grad);
    for (int k = 0; k < m_; k++) {
      v->Set_Entry(k, 0, u_std(k) - std::pow((*x_)(k, 0) + 1.0, 3.0));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> grad = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*grad, *v);
    for (int k = 0; k < m_; k++) {
      u_grad_std.Set_Entry(k, (*grad)(k, 0));
    }
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT>& u_out, const HDSA::Vector<RealT>& u_in, const HDSA::Vector<RealT>& u,
                            const HDSA::Vector<RealT>& z) const {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(u_in);
    HDSA::Std_Vector<RealT> u_out_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(u_out);
    for (int k = 0; k < m_; k++) {
      v->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Hv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*Hv, *v);
    for (int k = 0; k < m_; k++) {
      u_out_std.Set_Entry(k, (*Hv)(k, 0));
    }
  }

  void State_Solve(HDSA::Vector<RealT>& u_out, const HDSA::Vector<RealT>& z) const {
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT>&>(z);
    HDSA::Std_Vector<RealT> u_out_std = dynamic_cast<HDSA::Std_Vector<RealT>&>(u_out);
    for (int k = 0; k < m_; k++) {
      u_out_std.Set_Entry(k, std::pow(z_std(k), 3.0));
    }
  }

  void Regularization_Gradient(HDSA::Vector<RealT>& grad_z, const HDSA::Vector<RealT>& u,
                               const HDSA::Vector<RealT>& z) const {
    grad_z.Zeros();
  }
};

#endif
