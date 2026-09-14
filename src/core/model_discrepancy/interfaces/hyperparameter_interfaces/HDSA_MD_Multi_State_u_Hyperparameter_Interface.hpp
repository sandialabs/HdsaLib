/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_MULTI_STATE_U_HYPERPARAMETER_INTERFACE_HPP
#define HDSA_MD_MULTI_STATE_U_HYPERPARAMETER_INTERFACE_HPP

namespace HDSA
{

  template <class RealT>
  class MD_Multi_State_u_Hyperparameter_Interface : public HDSA::MD_u_Hyperparameter_Interface<RealT>
  {

  private:
    std::vector<HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>>> u_hyperparam_interface_std_;

  public:
    MD_Multi_State_u_Hyperparameter_Interface(std::vector<HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>>> &u_hyperparam_interface_std) : HDSA::MD_u_Hyperparameter_Interface<RealT>(u_hyperparam_interface_std[0]->Is_Transient()), u_hyperparam_interface_std_(u_hyperparam_interface_std)
    {
      HDSA::MD_u_Hyperparameter_Interface<RealT>::is_multistate_interface_ = true;
    }

    virtual ~MD_Multi_State_u_Hyperparameter_Interface()
    {
    }

    HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> Get_Hyperparameter_Interface_k(int k) const
    {
      return u_hyperparam_interface_std_[k];
    }

    int Get_Number_of_States(void) const
    {
      return u_hyperparam_interface_std_.size();
    }

    RealT Get_alpha_d(void) const override
    {
      RealT alpha_d = 0.0;
      for (long unsigned int k = 0; k < u_hyperparam_interface_std_.size(); k++)
      {
        alpha_d += u_hyperparam_interface_std_[k]->Get_alpha_d();
      }
      alpha_d /= static_cast<RealT>(u_hyperparam_interface_std_.size());
      return alpha_d;
    }
  };

}

#endif
