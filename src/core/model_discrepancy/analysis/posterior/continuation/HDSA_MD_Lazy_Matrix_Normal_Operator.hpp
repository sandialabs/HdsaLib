/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_LAZY_MATRIX_NORMAL_OPERATOR_HPP
#define HDSA_MD_LAZY_MATRIX_NORMAL_OPERATOR_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MultiVector.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Lazy_Matrix_Normal_Operator
  {
  private:
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> sigma_apply_;
    std::function<void(HDSA::MultiVector<RealT> &)> sigma_sample_;

    HDSA::Ptr<HDSA::Vector<RealT>> input_prototype_;
    HDSA::Ptr<HDSA::Vector<RealT>> output_prototype_;

    HDSA::Ptr<HDSA::MultiVector<RealT>> Q_l_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> W_Q_l_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Q_r_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Sigma_Q_r_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Y_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> T_;

    RealT tol_;

    HDSA::Ptr<HDSA::Vector<RealT>> Sample_Left_Residual(void)
    {
      HDSA::MultiVector<RealT> samples(1, *output_prototype_);
      u_prior_interface_->Sample_with_Covariance_W_u_Inverse(samples);
      HDSA::Ptr<HDSA::Vector<RealT>> eta = output_prototype_->Clone();
      eta->Set(*samples[0]);

      const int kl = Q_l_->Number_of_Vectors();
      if (kl == 0)
        return eta;

      for (int pass = 0; pass < 2; ++pass)
      {
        for (int j = 0; j < kl; ++j)
        {
          const RealT coeff = (*W_Q_l_)[j]->Dot(*eta);
          eta->Scaled_Plus(-coeff, *(*Q_l_)[j]);
        }
      }
      return eta;
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Sample_Right_Residual(void)
    {
      HDSA::MultiVector<RealT> samples(1, *input_prototype_);
      sigma_sample_(samples);
      HDSA::Ptr<HDSA::Vector<RealT>> eta = input_prototype_->Clone();
      eta->Set(*samples[0]);

      const int kr = Q_r_->Number_of_Vectors();
      if (kr == 0)
        return eta;

      for (int pass = 0; pass < 2; ++pass)
      {
        for (int j = 0; j < kr; ++j)
        {
          const RealT coeff = (*Q_r_)[j]->Dot(*eta);
          eta->Scaled_Plus(-coeff, *(*Sigma_Q_r_)[j]);
        }
      }
      return eta;
    }

  public:
    MD_Lazy_Matrix_Normal_Operator(
        const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface,
        const std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> &sigma_apply,
        const std::function<void(HDSA::MultiVector<RealT> &)> &sigma_sample,
        const HDSA::Vector<RealT> &input_prototype,
        const HDSA::Vector<RealT> &output_prototype,
        const RealT tol = static_cast<RealT>(1.e-10))
        : u_prior_interface_(u_prior_interface), sigma_apply_(sigma_apply), sigma_sample_(sigma_sample), tol_(tol)
    {
      input_prototype_ = input_prototype.Clone();
      output_prototype_ = output_prototype.Clone();
      Q_l_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
      W_Q_l_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
      Q_r_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
      Sigma_Q_r_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
      Y_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
      T_ = HDSA::makePtr<HDSA::MultiVector<RealT>>();
    }

    virtual ~MD_Lazy_Matrix_Normal_Operator()
    {
    }

    void Forward_Apply(HDSA::Vector<RealT> &y, const HDSA::Vector<RealT> &v)
    {
      HDSA_TEST_FOR_EXCEPTION(v.Dimension() != input_prototype_->Dimension(), std::logic_error,
                              "Error in HDSA::MD_Lazy_Matrix_Normal_Operator::Forward_Apply: input has wrong dimension." << std::endl);

      HDSA::Ptr<HDSA::Vector<RealT>> Sigma_v = input_prototype_->Clone();
      sigma_apply_(*Sigma_v, v);

      std::vector<RealT> c(Q_r_->Number_of_Vectors(), 0.0);
      HDSA::Ptr<HDSA::Vector<RealT>> r = input_prototype_->Clone();
      r->Set(v);
      HDSA::Ptr<HDSA::Vector<RealT>> Sigma_r = input_prototype_->Clone();
      Sigma_r->Set(*Sigma_v);

      const int kr = Q_r_->Number_of_Vectors();
      for (int j = 0; j < kr; ++j)
      {
        c[j] = (*Q_r_)[j]->Dot(*Sigma_v);
        r->Scaled_Plus(-c[j], *(*Q_r_)[j]);
        Sigma_r->Scaled_Plus(-c[j], *(*Sigma_Q_r_)[j]);
      }
      for (int j = 0; j < kr; ++j)
      {
        const RealT dc = (*Q_r_)[j]->Dot(*Sigma_r);
        c[j] += dc;
        r->Scaled_Plus(-dc, *(*Q_r_)[j]);
        Sigma_r->Scaled_Plus(-dc, *(*Sigma_Q_r_)[j]);
      }

      const RealT norm_r = std::sqrt(std::max(r->Dot(*Sigma_r), static_cast<RealT>(0.0)));
      if (norm_r > tol_)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> q_new = input_prototype_->Clone();
        q_new->Set(*r);
        q_new->Scale(static_cast<RealT>(1.0) / norm_r);

        HDSA::Ptr<HDSA::Vector<RealT>> Sigma_q_new = input_prototype_->Clone();
        Sigma_q_new->Set(*Sigma_r);
        Sigma_q_new->Scale(static_cast<RealT>(1.0) / norm_r);

        HDSA::Ptr<HDSA::Vector<RealT>> y_new = output_prototype_->Clone();
        y_new->Zeros();
        const int kl = Q_l_->Number_of_Vectors();
        for (int j = 0; j < kl; ++j)
        {
          const RealT coeff = (*T_)[j]->Dot(*q_new);
          y_new->Scaled_Plus(coeff, *(*Q_l_)[j]);
        }
        HDSA::Ptr<HDSA::Vector<RealT>> eta = Sample_Left_Residual();
        y_new->Plus(*eta);

        Q_r_->push_back(q_new);
        Sigma_Q_r_->push_back(Sigma_q_new);
        Y_->push_back(y_new);
        c.push_back(norm_r);
      }

      y.Zeros();
      const int ny = Y_->Number_of_Vectors();
      for (int j = 0; j < ny; ++j)
        y.Scaled_Plus(c[j], *(*Y_)[j]);
    }

    void Adjoint_Apply(HDSA::Vector<RealT> &y, const HDSA::Vector<RealT> &u)
    {
      HDSA_TEST_FOR_EXCEPTION(u.Dimension() != output_prototype_->Dimension(), std::logic_error,
                              "Error in HDSA::MD_Lazy_Matrix_Normal_Operator::Adjoint_Apply: input has wrong dimension." << std::endl);

      HDSA::Ptr<HDSA::Vector<RealT>> x = output_prototype_->Clone();
      u_prior_interface_->Apply_W_u_Inverse(*x, u);

      std::vector<RealT> c(Q_l_->Number_of_Vectors(), 0.0);
      HDSA::Ptr<HDSA::Vector<RealT>> r = output_prototype_->Clone();
      r->Set(*x);
      HDSA::Ptr<HDSA::Vector<RealT>> W_r = output_prototype_->Clone();
      W_r->Set(u);

      const int kl = Q_l_->Number_of_Vectors();
      for (int j = 0; j < kl; ++j)
      {
        c[j] = (*Q_l_)[j]->Dot(u);
        r->Scaled_Plus(-c[j], *(*Q_l_)[j]);
        W_r->Scaled_Plus(-c[j], *(*W_Q_l_)[j]);
      }
      for (int j = 0; j < kl; ++j)
      {
        const RealT dc = (*W_Q_l_)[j]->Dot(*r);
        c[j] += dc;
        r->Scaled_Plus(-dc, *(*Q_l_)[j]);
        W_r->Scaled_Plus(-dc, *(*W_Q_l_)[j]);
      }

      const RealT norm_r = std::sqrt(std::max(r->Dot(*W_r), static_cast<RealT>(0.0)));
      if (norm_r > tol_)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> q_new = output_prototype_->Clone();
        q_new->Set(*r);
        q_new->Scale(static_cast<RealT>(1.0) / norm_r);

        HDSA::Ptr<HDSA::Vector<RealT>> W_q_new = output_prototype_->Clone();
        W_q_new->Set(*W_r);
        W_q_new->Scale(static_cast<RealT>(1.0) / norm_r);

        HDSA::Ptr<HDSA::Vector<RealT>> t_new = input_prototype_->Clone();
        t_new->Zeros();
        const int kr = Q_r_->Number_of_Vectors();
        for (int j = 0; j < kr; ++j)
        {
          const RealT coeff = (*Y_)[j]->Dot(*W_q_new);
          t_new->Scaled_Plus(coeff, *(*Sigma_Q_r_)[j]);
        }
        HDSA::Ptr<HDSA::Vector<RealT>> eta = Sample_Right_Residual();
        t_new->Plus(*eta);

        Q_l_->push_back(q_new);
        W_Q_l_->push_back(W_q_new);
        T_->push_back(t_new);
        c.push_back(norm_r);
      }

      y.Zeros();
      const int nt = T_->Number_of_Vectors();
      for (int j = 0; j < nt; ++j)
        y.Scaled_Plus(c[j], *(*T_)[j]);
    }

    int Number_Left_Basis_Vectors(void) const { return Q_l_->Number_of_Vectors(); }
    int Number_Right_Basis_Vectors(void) const { return Q_r_->Number_of_Vectors(); }
  };

}

#endif
