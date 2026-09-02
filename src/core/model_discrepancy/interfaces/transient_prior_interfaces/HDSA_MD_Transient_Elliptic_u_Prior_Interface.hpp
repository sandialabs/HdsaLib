/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_TRANSIENT_ELLIPTIC_U_PRIOR_INTERFACE_HPP
#define HDSA_MD_TRANSIENT_ELLIPTIC_U_PRIOR_INTERFACE_HPP

#include "HDSA_MD_Transient_Prior_Covariance.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Transient_Elliptic_u_Prior_Interface : public HDSA::MD_Scaled_u_Prior_Interface<RealT>
  {

  private:
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> spatial_prior_cov_;
    HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<RealT>> transient_prior_cov_;
    HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> u_hyperparam_interface_;
    HDSA::Ptr<HDSA::MD_Determine_u_Hyperparameters<RealT>> determine_u_hyperparams_;
    int n_t_;

  public:
    void Apply_M_u(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const override
    {
      if (const HDSA::Transient_Vector<RealT> *u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&u_in))
      {
        HDSA::Transient_Vector<RealT> *u_out_trans = dynamic_cast<HDSA::Transient_Vector<RealT> *>(&u_out);
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = u_out.Clone();
        HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_tmp_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_tmp);
        for (int j = 0; j < n_t_; j++)
        {
          spatial_prior_cov_->Apply_M_u(*(*u_tmp_trans)[j], *u_in_trans->Get_Vector_Const(j));
        }
        for (int j = 0; j < n_t_; j++)
        {
          (*u_out_trans)[j]->Zeros();
          for (int i = 0; i < n_t_; i++)
          {
            (*u_out_trans)[j]->Scaled_Plus((*transient_prior_cov_->Get_M_t())(i, j), *(*u_tmp_trans)[i]);
          }
        }
      }
      else
      {
        spatial_prior_cov_->Apply_M_u(u_out, u_in);
      }
    }

    void Apply_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const override
    {
      const HDSA::Transient_Vector<RealT> *u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&u_in);
      HDSA::Transient_Vector<RealT> *u_out_trans = dynamic_cast<HDSA::Transient_Vector<RealT> *>(&u_out);

      if (HDSA::Ptr<const MD_Elliptic_u_Prior_Interface<RealT>> spatial_prior_cov_elliptic = HDSA::dynamicPtrCast<const MD_Elliptic_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_sing_vecs = spatial_prior_cov_elliptic->Get_Sing_Vecs_Output();
        int spatial_rank = spatial_sing_vecs->Number_of_Vectors();
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp1 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
        for (int i = 0; i < spatial_rank; i++)
        {
          for (int j = 0; j < n_t_; j++)
          {
            RealT val = (*spatial_sing_vecs)[i]->Dot(*u_in_trans->Get_Vector_Const(j));
            tmp1->Set_Entry(i, j, val);
          }
        }
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp2 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
        tmp1->Multiply(*tmp2, *transient_prior_cov_->Get_Evecs());

        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> spatial_sing_vals = spatial_prior_cov_elliptic->Get_Sing_Vals();
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_sing_vals = transient_prior_cov_->Get_Evals();
        for (int i = 0; i < spatial_rank; i++)
        {
          for (int j = 0; j < n_t_; j++)
          {
            RealT val1 = std::pow((*spatial_sing_vals)(i, 0), 2.0) * (*time_sing_vals)(j, 0);
            RealT val2 = val1 / (1.0 + scalar * val1);
            RealT val3 = (*tmp2)(i, j) * val2;
            tmp2->Set_Entry(i, j, val3);
          }
        }

        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp3 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
        tmp2->Multiply(*tmp3, *transient_prior_cov_->Get_Evecs(), false, true);
        for (int j = 0; j < n_t_; j++)
        {
          (*u_out_trans)[j]->Zeros();
          for (int i = 0; i < spatial_rank; i++)
          {
            (*u_out_trans)[j]->Scaled_Plus((*tmp3)(i, j), *(*spatial_sing_vecs)[i]);
          }
        }
      }
      else if (HDSA::Ptr<const MD_Lumped_Mass_u_Prior_Interface<RealT>> spatial_prior_cov_lump = HDSA::dynamicPtrCast<const MD_Lumped_Mass_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp1 = u_out.Clone();
        HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_tmp1_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_tmp1);
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = u_out.Clone();
        HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_tmp2_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_tmp2);
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evals = transient_prior_cov_->Get_Evals();
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evecs = transient_prior_cov_->Get_Evecs();
        for (int i = 0; i < n_t_; i++)
        {
          for (int j = 0; j < n_t_; j++)
          {
            RealT val = std::sqrt((*time_evals)(i, 0)) * (*time_evecs)(j, i);
            (*u_tmp1_trans)[i]->Scaled_Plus(val, *u_in_trans->Get_Vector_Const(j));
          }
          spatial_prior_cov_lump->Apply_W_u_Acute_Plus_scalar_M_u_Inverse(*(*u_tmp2_trans)[i], *(*u_tmp1_trans)[i], (*time_evals)(i, 0) * scalar);
        }

        for (int i = 0; i < n_t_; i++)
        {
          for (int j = 0; j < n_t_; j++)
          {
            RealT val = std::sqrt((*time_evals)(j, 0)) * (*time_evecs)(i, j);
            (*u_out_trans)[i]->Scaled_Plus(val, *u_tmp2_trans->Get_Vector_Const(j));
          }
        }
      }
    }

    void Apply_W_u_Acute_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const override
    {
      HDSA::Ptr<const MD_Scaled_u_Prior_Interface<RealT>> spatial_prior_cov_cast = HDSA::dynamicPtrCast<const MD_Scaled_u_Prior_Interface<RealT>>(spatial_prior_cov_);
      const HDSA::Transient_Vector<RealT> *u_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&u_in);
      HDSA::Transient_Vector<RealT> *u_out_trans = dynamic_cast<HDSA::Transient_Vector<RealT> *>(&u_out);
      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = u_out.Clone();
      HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_tmp_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_tmp);
      for (int j = 0; j < n_t_; j++)
      {
        spatial_prior_cov_cast->Apply_W_u_Acute_Inverse(*(*u_tmp_trans)[j], *u_in_trans->Get_Vector_Const(j));
      }
      for (int j = 0; j < n_t_; j++)
      {
        (*u_out_trans)[j]->Zeros();
        for (int i = 0; i < n_t_; i++)
        {
          (*u_out_trans)[j]->Scaled_Plus((*transient_prior_cov_->Get_W_t_Inverse())(i, j), *(*u_tmp_trans)[i]);
        }
      }
    }

    void Sample_with_Covariance_W_u_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const override
    {
      int num_samples = samples.Number_of_Vectors();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evals = transient_prior_cov_->Get_Evals();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evecs = transient_prior_cov_->Get_Evecs();

      if (HDSA::Ptr<const MD_Elliptic_u_Prior_Interface<RealT>> spatial_prior_cov_elliptic = HDSA::dynamicPtrCast<const MD_Elliptic_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = spatial_prior_cov_elliptic->Get_Random_Number_Generator();
        HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_sing_vecs = spatial_prior_cov_elliptic->Get_Sing_Vecs_Output();
        int spatial_rank = spatial_sing_vecs->Number_of_Vectors();
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> spatial_sing_vals = spatial_prior_cov_elliptic->Get_Sing_Vals();
        for (int k = 0; k < num_samples; k++)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> sk = samples[k];
          HDSA::Ptr<HDSA::Transient_Vector<RealT>> sk_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(sk);

          HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp1 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
          for (int j = 0; j < n_t_; j++)
          {
            for (int i = 0; i < spatial_rank; i++)
            {
              RealT omega = random_number_generator->Generate_Standard_Normal_Sample();
              RealT val = (*spatial_sing_vals)(i, 0) * omega * std::sqrt((*time_evals)(j, 0));
              tmp1->Set_Entry(i, j, val);
            }
          }
          HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp2 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
          tmp1->Multiply(*tmp2, *transient_prior_cov_->Get_Evecs(), false, true);
          for (int j = 0; j < n_t_; j++)
          {
            (*sk_trans)[j]->Zeros();
            for (int i = 0; i < spatial_rank; i++)
            {
              (*sk_trans)[j]->Scaled_Plus((*tmp2)(i, j), *(*spatial_sing_vecs)[i]);
            }
          }
        }
      }
      else if (HDSA::Ptr<const MD_Lumped_Mass_u_Prior_Interface<RealT>> spatial_prior_cov_lump = HDSA::dynamicPtrCast<const MD_Lumped_Mass_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        for (int k = 0; k < num_samples; k++)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> sk = samples[k];
          HDSA::Ptr<HDSA::Transient_Vector<RealT>> sk_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(sk);

          HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_vecs = HDSA::makePtr<HDSA::MultiVector<RealT>>(n_t_, *(*sk_trans)[0]);
          spatial_prior_cov_lump->Sample_with_Covariance_W_u_Acute_Inverse(*spatial_vecs);

          for (int i = 0; i < n_t_; i++)
          {
            for (int j = 0; j < n_t_; j++)
            {
              RealT val = std::sqrt((*time_evals)(j, 0)) * (*time_evecs)(i, j);
              (*sk_trans)[i]->Scaled_Plus(val, *(*spatial_vecs)[j]);
            }
          }
        }
      }
    }

    void Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const override
    {
      int num_samples = samples.Number_of_Vectors();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evals = transient_prior_cov_->Get_Evals();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evecs = transient_prior_cov_->Get_Evecs();

      if (HDSA::Ptr<const MD_Elliptic_u_Prior_Interface<RealT>> spatial_prior_cov_elliptic = HDSA::dynamicPtrCast<const MD_Elliptic_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = spatial_prior_cov_elliptic->Get_Random_Number_Generator();
        HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_sing_vecs = spatial_prior_cov_elliptic->Get_Sing_Vecs_Output();
        int spatial_rank = spatial_sing_vecs->Number_of_Vectors();
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> spatial_sing_vals = spatial_prior_cov_elliptic->Get_Sing_Vals();
        for (int k = 0; k < num_samples; k++)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> sk = samples[k];
          HDSA::Ptr<HDSA::Transient_Vector<RealT>> sk_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(sk);

          HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp1 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
          for (int j = 0; j < n_t_; j++)
          {
            for (int i = 0; i < spatial_rank; i++)
            {
              RealT omega = random_number_generator->Generate_Standard_Normal_Sample();
              RealT val1 = std::pow((*spatial_sing_vals)(i, 0), 2.0) * (*time_evals)(j, 0);
              RealT val2 = val1 / (1.0 + scalar * val1);
              RealT val = std::sqrt(val2) * omega;
              tmp1->Set_Entry(i, j, val);
            }
          }
          HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp2 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(spatial_rank, n_t_);
          tmp1->Multiply(*tmp2, *transient_prior_cov_->Get_Evecs(), false, true);
          for (int j = 0; j < n_t_; j++)
          {
            (*sk_trans)[j]->Zeros();
            for (int i = 0; i < spatial_rank; i++)
            {
              (*sk_trans)[j]->Scaled_Plus((*tmp2)(i, j), *(*spatial_sing_vecs)[i]);
            }
          }
        }
      }
      else if (HDSA::Ptr<const MD_Lumped_Mass_u_Prior_Interface<RealT>> spatial_prior_cov_lump = HDSA::dynamicPtrCast<const MD_Lumped_Mass_u_Prior_Interface<RealT>>(spatial_prior_cov_))
      {
        for (int k = 0; k < num_samples; k++)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> sk = samples[k];
          HDSA::Ptr<HDSA::Transient_Vector<RealT>> sk_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(sk);

          std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> spatial_vecs;
          spatial_vecs.resize(n_t_);
          for (int j = 0; j < n_t_; j++)
          {
            spatial_vecs[j] = HDSA::makePtr<HDSA::MultiVector<RealT>>(1, *(*sk_trans)[j]);
            spatial_prior_cov_lump->Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(*spatial_vecs[j], scalar * (*time_evals)(j, 0));
          }

          for (int i = 0; i < n_t_; i++)
          {
            for (int j = 0; j < n_t_; j++)
            {
              RealT val = std::sqrt((*time_evals)(j, 0)) * (*time_evecs)(i, j);
              (*sk_trans)[i]->Scaled_Plus(val, *(*spatial_vecs[j])[0]);
            }
          }
        }
      }
    }

    void Precompute_W_u_Plus_scalar_M_u_Data(RealT &scalar) override
    {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> time_evals = transient_prior_cov_->Get_Evals();
      for (int i = 0; i < n_t_; i++)
      {
        RealT val = scalar * (*time_evals)(i, 0);
        spatial_prior_cov_->Precompute_W_u_Plus_scalar_M_u_Data(val);
      }
    }

    MD_Transient_Elliptic_u_Prior_Interface(HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &spatial_prior_cov, HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<RealT>> &transient_prior_cov) : HDSA::MD_Scaled_u_Prior_Interface<RealT>(transient_prior_cov->Get_Hyperparameter_Interface()->Get_alpha_u())
    {
      spatial_prior_cov_ = spatial_prior_cov;
      transient_prior_cov_ = transient_prior_cov;
      u_hyperparam_interface_ = transient_prior_cov_->Get_Hyperparameter_Interface();
      determine_u_hyperparams_ = transient_prior_cov_->Get_Determine_Hyperparameters();
      n_t_ = transient_prior_cov_->Get_n_t();

      if (u_hyperparam_interface_->Adapt_Time_Variance())
      {
        determine_u_hyperparams_->Determine_alpha_t(this);
      }
      transient_prior_cov_->Set_alpha_t(u_hyperparam_interface_->Get_alpha_t());

      if (u_hyperparam_interface_->Get_alpha_u() == 0.0)
      {
        determine_u_hyperparams_->Determine_alpha_u(this);
      }
      this->Set_alpha_u(u_hyperparam_interface_->Get_alpha_u());
    }

    virtual ~MD_Transient_Elliptic_u_Prior_Interface()
    {
    }

    HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> Get_Spatial_Cov(void) const
    {
      return spatial_prior_cov_;
    }

    HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<RealT>> Get_Time_Cov(void) const
    {
      return transient_prior_cov_;
    }
  };
}

#endif
