/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_CONTINUATION_UPDATE_HPP
#define HDSA_MD_CONTINUATION_UPDATE_HPP

#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Continuation_Sensitivity_Operators.hpp"
#include "HDSA_MD_Quasi_Newton_Preconditioner.hpp"
#include "HDSA_MD_Discrepancy_Parameter_Trajectory.hpp"
#include "HDSA_PC_Pseudo_Time_Continuation.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Continuation_Update
  {

  private:
    HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling_;
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;
    HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface_;
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt_;
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt_;
    int num_continuation_steps_;
    int r_;

    void Posterior_Update_Core(HDSA::Vector<RealT> &u_k, HDSA::Vector<RealT> &z_k, HDSA::Vector<RealT> &beta_k, int sample_idx) const
    {
      HDSA::Ptr<HDSA::MD_Continuation_Sensitivity_Operators<RealT>> sen_op = 
          HDSA::makePtr<HDSA::MD_Continuation_Sensitivity_Operators<RealT>>(post_sampling_, hessian_analysis_);
      
      HDSA::Ptr<HDSA::MD_Quasi_Newton_Preconditioner<RealT>> qn_prec = 
          HDSA::makePtr<HDSA::MD_Quasi_Newton_Preconditioner<RealT>>(hessian_analysis_);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_nom = beta_k.Clone();
      beta_nom->Zeros();

      HDSA::Ptr<HDSA::PC_Sensitivity_Operator_Interface<RealT>> base_sen_op = Teuchos::rcp_dynamic_cast<HDSA::PC_Sensitivity_Operator_Interface<RealT>>(sen_op);
      HDSA::Ptr<HDSA::PC_Pseudo_Time_Continuation<RealT>> pt_cont = 
          HDSA::makePtr<HDSA::PC_Pseudo_Time_Continuation<RealT>>(beta_nom, base_sen_op, qn_prec);

      HDSA::MD_Discrepancy_Parameter_Trajectory<RealT> theta_traj(num_continuation_steps_, sample_idx);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_k_tmp = beta_nom->Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> grad_k_tmp = beta_nom->Clone();
      
      pt_cont->Pseudo_Time_Continuation_Forward_Euler(*beta_k_tmp, *grad_k_tmp, theta_traj);

      beta_k.Set(*beta_k_tmp);
      if(sen_op->Get_Current_U()!=HDSA::nullPtr) u_k.Set(*sen_op->Get_Current_U());
      if(sen_op->Get_Current_Z()!=HDSA::nullPtr) z_k.Set(*sen_op->Get_Current_Z());
    }

  public:
    MD_Continuation_Update(const HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> &post_sampling,
                           const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis,
                           int num_continuation_steps)
        : post_sampling_(post_sampling), hessian_analysis_(hessian_analysis), num_continuation_steps_(num_continuation_steps)
    {
      opt_prob_interface_ = hessian_analysis_->Get_Opt_Prob_Interface();
      u_opt_ = post_sampling_->data_interface->Get_u_opt()->Clone();
      z_opt_ = post_sampling_->data_interface->Get_z_opt()->Clone();

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = hessian_analysis_->Get_Evals();
      if (evals == HDSA::nullPtr || evals->Number_of_Rows() == 0) {
        std::cout << "Hessian evaluations not available. Defaulting to full." << std::endl;
        r_ = z_opt_->Dimension();
      }
      else {
        r_ = evals->Number_of_Rows();
      }
    }

    ~MD_Continuation_Update(void)
    {
    }

    void Posterior_Update_Mean(HDSA::Vector<RealT> &u_k, HDSA::Vector<RealT> &z_k, HDSA::Vector<RealT> &beta_k) const
    {
      Posterior_Update_Core(u_k, z_k, beta_k, 0);
    }

    void Posterior_Update_Samples(HDSA::MultiVector<RealT> &u_ks, HDSA::MultiVector<RealT> &z_ks, HDSA::MultiVector<RealT> &beta_ks) const
    {
      int num_samples = post_sampling_->post_data->num_samples;
      for (int sample_idx = 1; sample_idx <= num_samples; ++sample_idx)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> u_k = (*u_ks)[sample_idx - 1];
        HDSA::Ptr<HDSA::Vector<RealT>> z_k = (*z_ks)[sample_idx - 1];
        HDSA::Ptr<HDSA::Vector<RealT>> beta_k = (*beta_ks)[sample_idx - 1];
        
        Posterior_Update_Core(*u_k, *z_k, *beta_k, sample_idx);
      }
    }
  };

}

#endif