/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OED_HPP
#define HDSA_MD_OED_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "HDSA_Dense_Matrix.hpp"
#include "HDSA_Dense_Vector_Utils.hpp"
#include "HDSA_Linear_Algebra.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Stack_Trace.hpp"
#include "HDSA_Vector.hpp"

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_MD_Ellipsoid_Ball_SPG.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"

namespace HDSA {

template <class RealT> class MD_OED {
public:
  struct Offline_Data {
    int r;

    HDSA::Ptr<HDSA::MultiVector<RealT>> V;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Mz_V;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Wz_inv_Mz_V;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Mz_Wz_inv_Mz_V;

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Vt_Mz_V;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Vt_Mz_Wz_inv_Mz_V;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> lambda;

    Offline_Data()
        : r(0), V(HDSA::nullPtr), Mz_V(HDSA::nullPtr), Wz_inv_Mz_V(HDSA::nullPtr), Mz_Wz_inv_Mz_V(HDSA::nullPtr),
          Vt_Mz_V(HDSA::nullPtr), Vt_Mz_Wz_inv_Mz_V(HDSA::nullPtr), lambda(HDSA::nullPtr) {}

    bool Is_Initialized() const {
      return r > 0 && V != HDSA::nullPtr && Mz_V != HDSA::nullPtr && Wz_inv_Mz_V != HDSA::nullPtr &&
             Mz_Wz_inv_Mz_V != HDSA::nullPtr && Vt_Mz_V != HDSA::nullPtr && Vt_Mz_Wz_inv_Mz_V != HDSA::nullPtr &&
             lambda != HDSA::nullPtr;
    }
  };

  using Ellipsoid_Ball_Projection_Data = typename HDSA::MD_Ellipsoid_Ball_SPG<RealT>::Projection_Data;
  using SPG_Options = typename HDSA::MD_Ellipsoid_Ball_SPG<RealT>::SPG_Options;
  using SPG_Info = typename HDSA::MD_Ellipsoid_Ball_SPG<RealT>::SPG_Info;

  struct Seq_Design_Result {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> beta_new;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z_new;
    SPG_Info optimizer_info;

    Seq_Design_Result() : beta_new(HDSA::nullPtr), Z_new(HDSA::nullPtr), optimizer_info() {}
  };

private:
  using DV = HDSA::Dense_Vector_Utils<RealT>;
  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface_;
  HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;

  Offline_Data offline_data_;

  bool verbosity_;
  RealT covar_coeff_;

  struct G_Eigs_Data {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> G;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> g;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> mu;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Mg;
  };

  void Apply_M_z_MultiVector(HDSA::MultiVector<RealT>& out, const HDSA::MultiVector<RealT>& in) const {
    const int num_vecs = in.Number_of_Vectors();
    for (int k = 0; k < num_vecs; ++k) {
      z_prior_interface_->Apply_M_z(*out[k], *in[k]);
    }
  }

  void Apply_W_z_Inverse_MultiVector(HDSA::MultiVector<RealT>& out, const HDSA::MultiVector<RealT>& in) const {
    const int num_vecs = in.Number_of_Vectors();
    for (int k = 0; k < num_vecs; ++k) {
      z_prior_interface_->Apply_W_z_Inverse(*out[k], *in[k]);
    }
  }

  void Symmetrize(HDSA::Dense_Matrix<RealT>& A) const {
    const int m = A.Number_of_Rows();
    const int n = A.Number_of_Columns();
    for (int i = 0; i < m; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const RealT val = static_cast<RealT>(0.5) * (A(i, j) + A(j, i));
        A.Set_Entry(i, j, val);
        A.Set_Entry(j, i, val);
      }
    }
  }

  void Check_Offline_Data_Initialized(const std::string& method_name) const {
    HDSA_TEST_FOR_EXCEPTION(!offline_data_.Is_Initialized(), std::logic_error,
                            "Error in HDSA::MD_OED::" << method_name << ": "
                                                      << "Offline data have not been initialized. "
                                                      << "Call Offline_Computation() before evaluating OED quantities."
                                                      << std::endl);
  }

  void Check_Reduced_Vector_Length(const HDSA::Dense_Matrix<RealT>& x, const int expected_length,
                                   const std::string& name, const std::string& method_name) const {
    const int len = DV::Length(x);

    HDSA_TEST_FOR_EXCEPTION(len != expected_length, std::logic_error,
                            "Error in HDSA::MD_OED::" << method_name << ": " << name << " has length " << len
                                                      << ", but expected length " << expected_length << "."
                                                      << std::endl);
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Linear_Combination_MultiVector(const HDSA::MultiVector<RealT>& X,
                                                                const HDSA::Dense_Matrix<RealT>& coeffs,
                                                                const int coeff_col) const {
    const int num_vecs = X.Number_of_Vectors();
    HDSA::Ptr<HDSA::Vector<RealT>> out = X[0]->Clone();
    out->Zeros();
    for (int k = 0; k < num_vecs; ++k) {
      out->Scaled_Plus(coeffs(k, coeff_col), *X[k]);
    }
    return out;
  }

  G_Eigs_Data Compute_G_Eigs_Value_Only(const HDSA::Dense_Matrix<RealT>& beta) const {
    Check_Offline_Data_Initialized("Compute_G_Eigs_Value_Only");

    const int r = offline_data_.r;
    const int beta_len = DV::Length(beta);

    HDSA_TEST_FOR_EXCEPTION(beta_len % r != 0, std::logic_error,
                            "Error in HDSA::MD_OED::Compute_G_Eigs_Value_Only: "
                            "beta length must be divisible by the reduced dimension r."
                                << std::endl);

    const int N = beta_len / r + 1;

    G_Eigs_Data data;

    data.M = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(r, N);
    data.M->Zeros();
    DV::Set_Columns_Column_Major(*data.M, 1, beta);

    const HDSA::Dense_Matrix<RealT>& A = *offline_data_.Vt_Mz_Wz_inv_Mz_V;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> AM = A.Multiply(*data.M);
    data.G = data.M->Multiply(*AM, true, false);

    HDSA::Dense_Matrix<RealT> G_plus_one = *data.G + static_cast<RealT>(1);
    data.G->Assign(G_plus_one);
    Symmetrize(*data.G);

    data.g = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, N);
    data.mu = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, 1);

    HDSA::Linear_Algebra::Symmetric_Eig_Decomposition<RealT>(*data.G, *data.g, *data.mu);
    data.Mg = data.M->Multiply(*data.g);
    return data;
  }

  RealT Compute_Trace_Term(const RealT mu_i, const RealT alpha_d) const {
    const int lambda_len = DV::Length(*offline_data_.lambda);
    RealT trace_term = static_cast<RealT>(0);
    for (int j = 0; j < lambda_len; ++j) {
      const RealT lambda_j = DV::Get_Column_Major(*offline_data_.lambda, j);
      const RealT denom = lambda_j * (mu_i + alpha_d * lambda_j);
      trace_term += static_cast<RealT>(1) / denom;
    }
    return trace_term;
  }

public:
  MD_OED(const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>>& data_interface,
         const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>>& u_prior_interface,
         const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>>& z_prior_interface,
         const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>>& hessian_analysis)
      : data_interface_(data_interface), u_prior_interface_(u_prior_interface), z_prior_interface_(z_prior_interface),
        hessian_analysis_(hessian_analysis), offline_data_(), verbosity_(false), covar_coeff_(static_cast<RealT>(1)) {}

  virtual ~MD_OED() {}

  void Set_Covariance_Coefficient(const RealT& covar_coeff) { covar_coeff_ = covar_coeff; }
  RealT Get_Covariance_Coefficient() const { return covar_coeff_; }

  void Set_Verbosity(const bool verbosity) { verbosity_ = verbosity; }
  bool Get_Verbosity() const { return verbosity_; }

  const Offline_Data& Get_Offline_Data() const { return offline_data_; }
  int Get_Reduced_Dimension() const { return offline_data_.r; }

  Seq_Design_Result Generate_Seq_Optimal_Design(const HDSA::Dense_Matrix<RealT>& beta_0, const RealT& alpha_d,
                                                const HDSA::Dense_Matrix<RealT>& betas,
                                                const HDSA::Dense_Matrix<RealT>& beta_bar, const RealT& constr_radius,
                                                const SPG_Options& options = SPG_Options()) const {
    Check_Offline_Data_Initialized("Generate_Seq_Optimal_Design");

    const int r = offline_data_.r;
    const int beta0_len = DV::Length(beta_0);
    Check_Reduced_Vector_Length(beta_bar, r, "beta_bar", "Generate_Seq_Optimal_Design");

    const int p = beta0_len / r;

    Ellipsoid_Ball_Projection_Data projection_data =
        HDSA::MD_Ellipsoid_Ball_SPG<RealT>::Prepare_Projection(beta_bar, p, constr_radius, *offline_data_.Vt_Mz_V);

    auto objective = [this, alpha_d, p, beta0_len, &betas, &beta_bar](const HDSA::Dense_Matrix<RealT>& candidate,
                                                                      HDSA::Dense_Matrix<RealT>& grad) -> RealT {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> full_beta = DV::Concatenate(betas, candidate);

      HDSA::Dense_Matrix<RealT> tail_grad(beta0_len, 1);
      tail_grad.Zeros();

      const RealT val = Evaluate_OED_Objective_Seq(*full_beta, alpha_d, beta_bar, p, tail_grad);

      for (int i = 0; i < beta0_len; ++i) {
        DV::Set_Column_Major(grad, i, DV::Get_Column_Major(tail_grad, i));
      }

      return val;
    };

    Seq_Design_Result result;

    result.beta_new = HDSA::MD_Ellipsoid_Ball_SPG<RealT>::Minimize(objective, beta_0, projection_data,
                                                                   result.optimizer_info, options);

    result.Z_new = HDSA::makePtr<HDSA::MultiVector<RealT>>(p, *data_interface_->Get_z_opt());

    for (int col = 0; col < p; ++col) {
      HDSA::Ptr<HDSA::Vector<RealT>> z_j = (*result.Z_new)[col];

      z_j->Set(*data_interface_->Get_z_opt());

      for (int i = 0; i < r; ++i) {
        const RealT coeff = DV::Get_Column_Major(*result.beta_new, i + r * col);

        z_j->Scaled_Plus(coeff, *(*offline_data_.V)[i]);
      }
    }

    return result;
  }

  void Offline_Computation() {
    offline_data_ = Offline_Data();

    HDSA::Ptr<HDSA::MultiVector<RealT>> evecs = hessian_analysis_->Get_Evecs();
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = hessian_analysis_->Get_Evals();

    HDSA_TEST_FOR_EXCEPTION(evecs == HDSA::nullPtr, std::logic_error,
                            "Error in HDSA::MD_OED::Offline_Computation: "
                            "Hessian eigenvectors are not available. "
                            "Call MD_Hessian_Analysis::Compute_Hessian_GEVP before OED. "
                            "Full-space OED fallback is not implemented in this implementation."
                                << std::endl);

    offline_data_.V = evecs;
    offline_data_.r = evecs->Number_of_Vectors();
    const int r = offline_data_.r;
    const HDSA::Vector<RealT>& prototype = *(*offline_data_.V)[0];

    offline_data_.Mz_V = HDSA::makePtr<HDSA::MultiVector<RealT>>(r, prototype);
    offline_data_.Wz_inv_Mz_V = HDSA::makePtr<HDSA::MultiVector<RealT>>(r, prototype);
    offline_data_.Mz_Wz_inv_Mz_V = HDSA::makePtr<HDSA::MultiVector<RealT>>(r, prototype);

    Apply_M_z_MultiVector(*offline_data_.Mz_V, *offline_data_.V);
    Apply_W_z_Inverse_MultiVector(*offline_data_.Wz_inv_Mz_V, *offline_data_.Mz_V);
    Apply_M_z_MultiVector(*offline_data_.Mz_Wz_inv_Mz_V, *offline_data_.Wz_inv_Mz_V);

    offline_data_.Vt_Mz_V = offline_data_.Mz_V->MatMat(*offline_data_.V);
    Symmetrize(*offline_data_.Vt_Mz_V);

    offline_data_.Vt_Mz_Wz_inv_Mz_V = offline_data_.Wz_inv_Mz_V->MatMat(*offline_data_.Mz_V);
    Symmetrize(*offline_data_.Vt_Mz_Wz_inv_Mz_V);

    offline_data_.lambda = u_prior_interface_->Get_W_u_Generalized_Eigenvalues();
  }

  RealT Evaluate_Posterior_Cov_Trace(const HDSA::Dense_Matrix<RealT>& beta, const RealT& alpha_d,
                                     const HDSA::Dense_Matrix<RealT>& beta_bar, HDSA::Dense_Matrix<RealT>& grad) const {
    Check_Offline_Data_Initialized("Evaluate_Posterior_Cov_Trace");

    const int r = offline_data_.r;
    const int beta_len = DV::Length(beta);

    HDSA_TEST_FOR_EXCEPTION(beta_len % r != 0, std::logic_error,
                            "Error in HDSA::MD_OED::Evaluate_Posterior_Cov_Trace: "
                            "beta length must be divisible by the reduced dimension r."
                                << std::endl);

    Check_Reduced_Vector_Length(beta_bar, r, "beta_bar", "Evaluate_Posterior_Cov_Trace");

    G_Eigs_Data geigs = Compute_G_Eigs_Value_Only(beta);

    const int N = beta_len / r + 1;
    const HDSA::Dense_Matrix<RealT>& A = *offline_data_.Vt_Mz_Wz_inv_Mz_V;
    const HDSA::Dense_Matrix<RealT>& M = *geigs.M;

    HDSA::Dense_Matrix<RealT> grad_local(beta.Number_of_Rows(), beta.Number_of_Columns());
    grad_local.Zeros();

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A_beta_bar = A.Multiply(beta_bar, true, false);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> MtA = M.Multiply(A, true, false);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A_Mg = A.Multiply(*geigs.Mg);

    const RealT max_abs_mu = DV::Max_Abs(*geigs.mu);
    const RealT mu_tol = static_cast<RealT>(1e-12) * std::max(static_cast<RealT>(1), max_abs_mu);
    const int lambda_len = DV::Length(*offline_data_.lambda);

    RealT val = static_cast<RealT>(0);

    for (int eig_idx = 0; eig_idx < N; ++eig_idx) {
      const RealT mu_i = (*geigs.mu)(eig_idx, 0);
      RealT trace_term = static_cast<RealT>(0);
      RealT d_trace_d_mu = static_cast<RealT>(0);

      for (int j = 0; j < lambda_len; ++j) {
        const RealT lambda_j = DV::Get_Column_Major(*offline_data_.lambda, j);
        const RealT denom = mu_i + alpha_d * lambda_j;
        trace_term += static_cast<RealT>(1) / (lambda_j * denom);
        d_trace_d_mu -= static_cast<RealT>(1) / (lambda_j * denom * denom);
      }

      HDSA::Ptr<HDSA::Vector<RealT>> tmp =
          Linear_Combination_MultiVector(*offline_data_.Mz_Wz_inv_Mz_V, *geigs.Mg, eig_idx);

      HDSA::Ptr<HDSA::Vector<RealT>> Wz_inv_tmp = tmp->Clone();
      z_prior_interface_->Apply_W_z_Inverse(*Wz_inv_tmp, *tmp);
      const RealT y_P_y = covar_coeff_ * tmp->Dot(*Wz_inv_tmp);

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> q = offline_data_.Mz_Wz_inv_Mz_V->MatVec(*Wz_inv_tmp);
      q->Scale(covar_coeff_);

      const RealT s_i = DV::Column_Sum(*geigs.g, eig_idx) + DV::Column_Dot(beta_bar, *A_Mg, eig_idx);
      const RealT p_i = s_i * s_i + y_P_y;
      val += p_i * trace_term;

      HDSA::Dense_Matrix<RealT> inv_denom(N, 1);
      inv_denom.Zeros();

      for (int k = 0; k < N; ++k) {
        const RealT denom = mu_i - (*geigs.mu)(k, 0);
        if (std::abs(denom) > mu_tol) { inv_denom.Set_Entry(k, 0, static_cast<RealT>(1) / denom); }
      }

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> g_scaled = DV::Scale_Columns(*geigs.g, inv_denom);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> mat = g_scaled->Multiply(*geigs.g, false, true);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> mat2 = mat->Multiply(*MtA);

      for (int design_col = 0; design_col < N - 1; ++design_col) {
        const RealT g_tail = (*geigs.g)(design_col + 1, eig_idx);
        for (int reduced_row = 0; reduced_row < r; ++reduced_row) {
          const int beta_idx = reduced_row + r * design_col;
          const RealT vec2_reduced_row = (*A_Mg)(reduced_row, eig_idx);
          const RealT mu_jac = static_cast<RealT>(2) * g_tail * vec2_reduced_row;

          HDSA::Dense_Matrix<RealT> g_jac_col =
              mat->Get_Column(design_col + 1) * vec2_reduced_row + mat2->Get_Column(reduced_row) * g_tail;

          HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Mg_jac = M.Multiply(g_jac_col);
          Mg_jac->Set_Entry(reduced_row, 0, (*Mg_jac)(reduced_row, 0) + g_tail);

          const RealT grad_s_i = DV::Column_Sum(g_jac_col, 0) + DV::Column_Dot(*Mg_jac, *A_beta_bar, 0);
          const RealT grad_y_P_y_i = static_cast<RealT>(2) * DV::Column_Dot(*Mg_jac, *q, 0);
          const RealT grad_p_i = static_cast<RealT>(2) * s_i * grad_s_i + grad_y_P_y_i;
          const RealT contribution = grad_p_i * trace_term + p_i * d_trace_d_mu * mu_jac;
          const RealT old_val = DV::Get_Column_Major(grad_local, beta_idx);
          DV::Set_Column_Major(grad_local, beta_idx, old_val + contribution);
        }
      }
    }

    HDSA_TEST_FOR_EXCEPTION(!std::isfinite(val), std::logic_error,
                            "Error in HDSA::MD_OED::Evaluate_Posterior_Cov_Trace: "
                            "Computed nonfinite OED objective value."
                                << std::endl);

    grad.Assign(grad_local);
    return val;
  }

  RealT Evaluate_OED_Objective_Seq(const HDSA::Dense_Matrix<RealT>& beta, const RealT& alpha_d,
                                   const HDSA::Dense_Matrix<RealT>& beta_bar, const int& p,
                                   HDSA::Dense_Matrix<RealT>& grad) const {
    Check_Offline_Data_Initialized("Evaluate_OED_Objective_Seq");

    const int r = offline_data_.r;
    HDSA_TEST_FOR_EXCEPTION(DV::Length(beta) % r != 0, std::logic_error,
                            "Error in HDSA::MD_OED::Evaluate_OED_Objective_Seq: "
                            "beta length must be divisible by reduced dimension r."
                                << std::endl);
    HDSA::Dense_Matrix<RealT> full_grad(beta.Number_of_Rows(), beta.Number_of_Columns());
    const RealT val_full = Evaluate_Posterior_Cov_Trace(beta, alpha_d, beta_bar, full_grad);
    grad.Assign(static_cast<RealT>(-1) * *(DV::Tail(full_grad, p * r)));
    return -val_full;
  }
};

} // namespace HDSA

#endif