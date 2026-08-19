/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_ELLIPTIC_U_PRIOR_INTERFACE_HPP
#define HDSA_MD_ELLIPTIC_U_PRIOR_INTERFACE_HPP

#include <algorithm>
#include <cstdlib>
#include <random>
#include "HDSA_MD_Scaled_u_Prior_Interface.hpp"
#include "HDSA_Randomized_GSVD.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Linear_Algebra.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Elliptic_u_Prior_Interface : public HDSA::MD_Scaled_u_Prior_Interface<RealT>
  {

  private:
    HDSA::Ptr<HDSA::Randomized_GSVD<RealT>> gsvd_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> sing_vecs_input_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> sing_vecs_output_;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> sing_vals_;
    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;

  public:
    MD_Elliptic_u_Prior_Interface(RealT alpha_u) : HDSA::MD_Scaled_u_Prior_Interface<RealT>(alpha_u), random_number_generator_(HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>())
    {
    }

    MD_Elliptic_u_Prior_Interface(RealT alpha_u, int seed) : HDSA::MD_Scaled_u_Prior_Interface<RealT>(alpha_u), random_number_generator_(HDSA::Random_Number_Generator<RealT>(seed))
    {
    }

    MD_Elliptic_u_Prior_Interface(RealT alpha_u, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator) : HDSA::MD_Scaled_u_Prior_Interface<RealT>(alpha_u), random_number_generator_(random_number_generator)
    {
    }

    virtual ~MD_Elliptic_u_Prior_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_E_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const = 0;

    virtual void Apply_E_u_Inverse_Transpose(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const = 0;

    virtual void Apply_M_u(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Accessor functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<HDSA::MultiVector<RealT>> Get_Sing_Vecs_Output(void) const
    {
      return sing_vecs_output_;
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_Sing_Vals(void) const
    {
      return sing_vals_;
    }

    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> Get_Random_Number_Generator(void) const
    {
      return random_number_generator_;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // User interface to GSVD
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Compute_E_u_Inverse_GSVD(int num_sing_vals, int oversampling, int num_subspace_iters, const HDSA::Vector<RealT> &u_vec)
    {
      gsvd_ = HDSA::makePtr<MD_Elliptic_GSVD<RealT>>(this);
      sing_vecs_output_ = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_sing_vals, u_vec);
      sing_vecs_input_ = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_sing_vals, u_vec);
      sing_vals_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(num_sing_vals, 1);
      gsvd_->Compute_GSVD(*sing_vecs_input_, *sing_vecs_output_, *sing_vals_, num_sing_vals, oversampling, num_subspace_iters);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of base class Virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const
    {
      u_out.Zeros();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> c = sing_vecs_output_->MatVec(u_in);
      for (int k = 0; k < c->Number_of_Rows(); k++)
      {
        RealT coeff = (*c)(k, 0) * std::pow((*sing_vals_)(k, 0), 2.0) / (1.0 + scalar * std::pow((*sing_vals_)(k, 0), 2.0));
        u_out.Scaled_Plus(coeff, *(*sing_vecs_output_)[k]);
      }
    }

    virtual void Apply_W_u_Acute_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
    {
      u_out.Zeros();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> c = sing_vecs_output_->MatVec(u_in);
      for (int k = 0; k < c->Number_of_Rows(); k++)
      {
        RealT coeff = (*c)(k, 0) * std::pow((*sing_vals_)(k, 0), 2.0);
        u_out.Scaled_Plus(coeff, *(*sing_vecs_output_)[k]);
      }
    }

    virtual HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_W_u_Generalized_Eigenvalues() const {
      const RealT alpha_u = this->Get_alpha_u();
      const int num_sing_vals = sing_vals_->Number_of_Rows();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> lambda_js = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(num_sing_vals, 1);
      for (int j = 0; j < num_sing_vals; ++j) {
        const RealT sigma_j = (*sing_vals_)(j, 0);
        HDSA_TEST_FOR_EXCEPTION(sigma_j == static_cast<RealT>(0), std::logic_error,
                                "Error in HDSA::MD_Elliptic_u_Prior_Interface::Get_W_u_Generalized_Eigenvalues: "
                                "Encountered zero singular value."
                                    << std::endl);
        lambda_js->Set_Entry(j, 0, static_cast<RealT>(1) / (alpha_u * std::pow(sigma_j, 2.0)));
      }
      return lambda_js;
    }

    // Compute samples from a mean zero Gaussian with covariance W_u^{-1}
    virtual void Sample_with_Covariance_W_u_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      int num_samples = samples.Number_of_Vectors();
      for (int k = 0; k < num_samples; k++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> vec = samples[k];
        for (int i = 0; i < sing_vals_->Number_of_Rows(); i++)
        {
          RealT rand = random_number_generator_->Generate_Standard_Normal_Sample();
          RealT coeff = (*sing_vals_)(i, 0) * rand;
          vec->Scaled_Plus(coeff, *(*sing_vecs_output_)[i]);
        }
      }
    }

    // Compute samples from a mean zero Gaussian with covariance W_u^{-1}
    virtual void Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const
    {
      int num_samples = samples.Number_of_Vectors();
      for (int k = 0; k < num_samples; k++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> vec = samples[k];
        for (int i = 0; i < sing_vals_->Number_of_Rows(); i++)
        {
          RealT rand = random_number_generator_->Generate_Standard_Normal_Sample();
          RealT coeff = rand * std::sqrt(std::pow((*sing_vals_)(i, 0), 2.0) / (1.0 + scalar * std::pow((*sing_vals_)(i, 0), 2.0)));
          vec->Scaled_Plus(coeff, *(*sing_vecs_output_)[i]);
        }
      }
    }

    template <class ScalarType>
    class MD_Elliptic_GSVD : public HDSA::Randomized_GSVD<ScalarType>
    {

    private:
      HDSA::MD_Elliptic_u_Prior_Interface<ScalarType> *u_prior_interface_;

    public:
      MD_Elliptic_GSVD(HDSA::MD_Elliptic_u_Prior_Interface<ScalarType> *u_prior_interface) : HDSA::Randomized_GSVD<ScalarType>()
      {
        u_prior_interface_ = u_prior_interface;
      }

      virtual ~MD_Elliptic_GSVD()
      {
      }

      void Apply_Operator(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        u_prior_interface_->Apply_E_u_Inverse(vec_out, vec_in);
      }

      void Apply_Operator_Transpose(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        u_prior_interface_->Apply_E_u_Inverse_Transpose(vec_out, vec_in);
      }

      void Apply_Input_Weighting_Operator_Inverse(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        u_prior_interface_->Apply_M_u(vec_out, vec_in);
      }

      void Apply_Output_Weighting_Operator(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        u_prior_interface_->Apply_M_u(vec_out, vec_in);
      }

      void Generate_Random_Samples(HDSA::MultiVector<RealT> &samples) const
      {
        samples.Randomize_Standard_Normal();
      }
    };
  };

}

#endif
