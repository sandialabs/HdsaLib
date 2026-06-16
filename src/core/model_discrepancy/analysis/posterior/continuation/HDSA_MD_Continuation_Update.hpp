/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_CONTINUATION_UPDATE_HPP
#define HDSA_MD_CONTINUATION_UPDATE_HPP

namespace HDSA {

template <class RealT> class MD_Continuation_Update {

private:
  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface_;
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface_;
  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling_;
  HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;
  int num_continuation_steps_;
  int r_;

public:
  MD_Continuation_Update(
      const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface,
      const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface,
      const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> &z_prior_interface,
      const HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> &opt_prob_interface,
      const HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> &post_sampling,
      const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis,
      int num_continuation_steps)
      : data_interface_(data_interface), u_prior_interface_(u_prior_interface),
        z_prior_interface_(z_prior_interface),
        opt_prob_interface_(opt_prob_interface), post_sampling_(post_sampling),
        hessian_analysis_(hessian_analysis), num_continuation_steps_(num_continuation_steps) {
          r_ = hessian_analysis_->Get_Evals()->Number_of_Rows(); // TODO: Check whether number of rows or columns. 
  }

  ~MD_Continuation_Update(void) {}

  HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>>
  Posterior_Update_Samples(void) const {
    HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_samples =
        HDSA::makePtr<HDSA::MD_Posterior_Vectors<RealT>>(
            post_sampling_->post_data->num_samples,
            *data_interface_->Get_z_opt());
    HDSA::Ptr<HDSA::Vector<RealT>> z_update_mean = Posterior_Update_Mean();
    posterior_samples->mean->Set(*z_update_mean);

    //////////////////// B_theta_hat
    HDSA::Ptr<HDSA::MultiVector<RealT>> u_tmp1 =
        HDSA::makePtr<HDSA::MultiVector<RealT>>(
            post_sampling_->post_data->num_samples,
            *data_interface_->Get_u_opt());
    HDSA::Ptr<HDSA::MultiVector<RealT>> B_theta_hat =
        HDSA::makePtr<HDSA::MultiVector<RealT>>(
            post_sampling_->post_data->num_samples,
            *data_interface_->Get_z_opt());
    for (int i = 0; i < post_sampling_->post_data->N; i++) {
      RealT coeff1 = post_sampling_->post_data->sum_g_vecs[i] /
                     std::sqrt((*post_sampling_->post_data->Mu)(i, 0));
      u_tmp1->Scaled_Plus(coeff1, *post_sampling_->post_data->u_i_hat[i]);

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> coeff2 =
          post_sampling_->post_data->u_i_hat[i]->MatVec(*state_grad_);

      // Compute M_z_W_z_inv_M_z_yi
      HDSA::Ptr<HDSA::Vector<RealT>> M_z_W_z_inv_M_z_yi =
          data_interface_->Get_z_opt()->Clone();
      M_z_W_z_inv_M_z_yi->Scaled_Plus(
          -post_sampling_->post_data->sum_g_vecs[i],
          *post_sampling_->post_data->M_z_W_z_inv_M_z_z_opt);
      for (int j = 0; j < post_sampling_->post_data->N; j++) {
        M_z_W_z_inv_M_z_yi->Scaled_Plus(
            (*post_sampling_->post_data->g_vecs)(j, i),
            *(*post_sampling_->post_data->M_z_W_z_inv_M_z_Z)[j]);
      }

      for (int k = 0; k < post_sampling_->post_data->num_samples; k++) {
        RealT val =
            (*coeff2)(k, 0) / std::sqrt((*post_sampling_->post_data->Mu)(i, 0));
        (*B_theta_hat)[k]->Scaled_Plus(val, *M_z_W_z_inv_M_z_yi);
      }
    }

    for (int k = 0; k < post_sampling_->post_data->num_samples; k++) {
      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 =
          data_interface_->Get_u_opt()->Clone();
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *(*u_tmp1)[k],
                                                *data_interface_->Get_u_opt(),
                                                *data_interface_->Get_z_opt());

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 =
          data_interface_->Get_z_opt()->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(
          *z_tmp1, *u_tmp2, *data_interface_->Get_z_opt());

      (*B_theta_hat)[k]->Plus(*z_tmp1);
    }

    B_theta_hat->Scale(std::sqrt(post_sampling_->post_data->alpha_d));

    //////////////// B_theta_breve
    HDSA::Ptr<HDSA::MultiVector<RealT>> B_theta_breve =
        HDSA::makePtr<HDSA::MultiVector<RealT>>(
            post_sampling_->post_data->num_samples,
            *data_interface_->Get_z_opt());

    if (post_sampling_->post_data->N > 1) {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp1 =
          post_sampling_->post_data->M_z_z_breve->MatMat(
              *post_sampling_->post_data
                   ->Zc); // Dimension (N-1)x(num_post_samples)

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp2 =
          HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(
              post_sampling_->post_data->N - 1,
              post_sampling_->post_data->num_samples);
      HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(
          *post_sampling_->post_data->Zc_M_z_W_z_inv_M_z_Zc, *tmp2, *tmp1);

      for (int k = 0; k < post_sampling_->post_data->num_samples; k++) {
        for (int i = 0; i < post_sampling_->post_data->N - 1; i++) {
          (*B_theta_breve)[k]->Scaled_Plus(
              -(*tmp2)(i, k),
              *(*post_sampling_->post_data->M_z_W_z_inv_M_z_Zc)[i]);
        }
      }
    }

    B_theta_breve->Scaled_Plus(1.0, *post_sampling_->post_data->M_z_z_breve);
    B_theta_breve->Scale(std::sqrt(state_grad_W_u_inv_state_grad_));

    HDSA::Ptr<HDSA::MultiVector<RealT>> z_update_samples =
        posterior_samples->samples;
    for (int k = 0; k < post_sampling_->post_data->num_samples; k++) {
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 =
          data_interface_->Get_z_opt()->Clone();
      z_tmp2->Set(*(*B_theta_hat)[k]);
      z_tmp2->Plus(*(*B_theta_breve)[k]);
      hessian_analysis_->Apply_RS_Hessian_Inverse(
          *(*z_update_samples)[k], *z_tmp2, *data_interface_->Get_z_opt());
      (*z_update_samples)[k]->Scale(-1.0);
      (*z_update_samples)[k]->Plus(*z_update_mean);
    }

    return posterior_samples;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Posterior_Update_Mean(void) const {
    HDSA::Ptr<HDSA::Vector<RealT>> z_update_mean =
        data_interface_->Get_z_opt()->Clone();

    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp1 = state_grad_->Clone();
    for (int ell = 0; ell < post_sampling_->post_data->N; ell++) {
      u_tmp1->Plus(*(*post_sampling_->post_data->u_ell)[ell]);
      for (int i = 0; i < post_sampling_->post_data->N; i++) {
        RealT coeff = post_sampling_->post_data->sum_g_vecs[i] *
                      (*post_sampling_->post_data->b_i_ell)(i, ell);
        u_tmp1->Scaled_Plus(-coeff,
                            *(*post_sampling_->post_data->u_i_ell[i])[ell]);
      }
    }
    u_tmp1->Scaled_Plus(post_sampling_->post_data->alpha_d,
                        *data_interface_->Get_data_shift());
    HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = u_tmp1->Clone();
    opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *u_tmp1,
                                              *data_interface_->Get_u_opt(),
                                              *data_interface_->Get_z_opt());

    HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_update_mean->Clone();
    opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(
        *z_tmp1, *u_tmp2, *data_interface_->Get_z_opt());

    for (int ell = 0; ell < post_sampling_->post_data->N; ell++) {
      RealT coeff1 =
          state_grad_->Dot(*(*post_sampling_->post_data->u_ell)[ell]);
      z_tmp1->Scaled_Plus(
          coeff1, *(*post_sampling_->post_data->M_z_W_z_inv_M_z_Z)[ell]);
      z_tmp1->Scaled_Plus(-coeff1,
                          *post_sampling_->post_data->M_z_W_z_inv_M_z_z_opt);
      for (int i = 0; i < post_sampling_->post_data->N; i++) {
        RealT coeff2 =
            state_grad_->Dot(*(*post_sampling_->post_data->u_i_ell[i])[ell]);
        coeff2 *= (*post_sampling_->post_data->b_i_ell)(i, ell);

        z_tmp1->Scaled_Plus(coeff2 * post_sampling_->post_data->sum_g_vecs[i],
                            *post_sampling_->post_data->M_z_W_z_inv_M_z_z_opt);
        for (int j = 0; j < post_sampling_->post_data->N; j++) {
          z_tmp1->Scaled_Plus(
              -coeff2 * (*post_sampling_->post_data->g_vecs)(j, i),
              *(*post_sampling_->post_data->M_z_W_z_inv_M_z_Z)[j]);
        }
      }
    }

    z_tmp1->Scale(-1.0 / post_sampling_->post_data->alpha_d);
    hessian_analysis_->Apply_RS_Hessian_Inverse(*z_update_mean, *z_tmp1,
                                                *data_interface_->Get_z_opt());
    z_update_mean->Plus(*data_interface_->Get_z_opt());

    return z_update_mean;
  }
};

} // namespace HDSA

#endif
