/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_VECTOR_Z_PRIOR_INTERFACE_HPP
#define HDSA_MD_VECTOR_Z_PRIOR_INTERFACE_HPP

#include "HDSA_MD_Scaled_z_Prior_Interface.hpp"
#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_Determine_z_Hyperparameters_Decl.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Vector_z_Prior_Interface : public HDSA::MD_Scaled_z_Prior_Interface<RealT>
  {

  private:
    const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
    const HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> z_hyperparam_interface_;
    const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
    HDSA::Ptr<HDSA::MD_Determine_z_Hyperparameters<RealT>> determine_z_hyperparams_;

  public:
    MD_Vector_z_Prior_Interface(const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, const HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> &z_hyperparam_interface, const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface) : HDSA::MD_Scaled_z_Prior_Interface<RealT>(z_hyperparam_interface->Get_alpha_z()), data_interface_(data_interface), z_hyperparam_interface_(z_hyperparam_interface), u_prior_interface_(u_prior_interface)
    {
      determine_z_hyperparams_ = HDSA::makePtr<HDSA::MD_Determine_z_Hyperparameters<RealT>>(data_interface_, z_hyperparam_interface_, u_prior_interface_);

      if (z_hyperparam_interface_->Get_alpha_z() == 0.0)
      {
        determine_z_hyperparams_->Determine_alpha_z(this);
      }
      this->Set_alpha_z(z_hyperparam_interface_->Get_alpha_z());
    }

    virtual ~MD_Vector_z_Prior_Interface()
    {
    }

    void Apply_M_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      z_out.Set(z_in);
    }

    void Apply_W_z_Acute_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      z_out.Set(z_in);
    }

    void Sample_with_Covariance_W_z_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      samples.Randomize_Standard_Normal();
    }

    void Apply_W_z_Acute(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      z_out.Set(z_in);
    }
  };

}

#include "HDSA_MD_Determine_z_Hyperparameters_Def.hpp"

#endif
