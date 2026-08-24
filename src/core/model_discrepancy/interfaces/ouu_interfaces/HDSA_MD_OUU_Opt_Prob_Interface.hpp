/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OUU_OPT_PROB_INTERFACE_HPP
#define HDSA_MD_OUU_OPT_PROB_INTERFACE_HPP

#include "HDSA_MD_Opt_Prob_Interface.hpp"
#include "HDSA_Ensemble_Vector.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_OUU_Opt_Prob_Interface : public HDSA::MD_Opt_Prob_Interface<RealT>
  {

  private:
    std::vector<RealT> ens_weights_;
    int ens_size_;

  public:
    MD_OUU_Opt_Prob_Interface(std::vector<RealT> &ens_weights)
    {
      ens_weights_ = ens_weights;
      ens_size_ = ens_weights.size();
    }

    MD_OUU_Opt_Prob_Interface(int ens_size)
    {
      ens_weights_ = std::vector<RealT>(ens_size, 1.0 / static_cast<RealT>(ens_size));
      ens_size_ = ens_size;
    }

    virtual ~MD_OUU_Opt_Prob_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, int s) const = 0;

    virtual void Apply_RS_Hessian_Per_Sample(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, int s) const = 0;

    virtual void Misfit_Gradient_Per_Sample(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z, int s) const = 0;

    virtual void Apply_Misfit_Hessian_Per_Sample(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z, int s) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Base class pure virtual function implementations
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const
    {
      z_out.Zeros();
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z_out.Clone();

      const HDSA::Ensemble_Vector<RealT> u_in_ens = dynamic_cast<const HDSA::Ensemble_Vector<RealT> &>(u_in);

      for (int s = 0; s < ens_size_; s++)
      {
        z_tmp->Zeros();
        Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(*z_tmp, *u_in_ens[s], z, s);
        z_out.Plus(*z_tmp);
      }
    }

    void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
    {
      z_out.Zeros();
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z_out.Clone();
      for (int s = 0; s < ens_size_; s++)
      {
        z_tmp->Zeros();
        Apply_RS_Hessian_Per_Sample(*z_tmp, z_in, z, s);
        z_out.Scaled_Plus(ens_weights_[s], *z_tmp);
      }
    }

    void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
    {
      const HDSA::Ensemble_Vector<RealT> u_ens = dynamic_cast<const HDSA::Ensemble_Vector<RealT> &>(u);
      HDSA::Ensemble_Vector<RealT> u_grad_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(u_grad);
      for (int s = 0; s < ens_size_; s++)
      {
        Misfit_Gradient_Per_Sample(*u_grad_ens[s], *u_ens[s], z, s);
        u_grad_ens[s]->Scale(ens_weights_[s]);
      }
    }

    void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
    {
      const HDSA::Ensemble_Vector<RealT> u_ens = dynamic_cast<const HDSA::Ensemble_Vector<RealT> &>(u);
      const HDSA::Ensemble_Vector<RealT> u_in_ens = dynamic_cast<const HDSA::Ensemble_Vector<RealT> &>(u_in);
      HDSA::Ensemble_Vector<RealT> u_out_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(u_out);
      for (int s = 0; s < ens_size_; s++)
      {
        Apply_Misfit_Hessian_Per_Sample(*u_out_ens[s], *u_in_ens[s], *u_ens[s], z, s);
        u_out_ens[s]->Scale(ens_weights_[s]);
      }
    }
  };

}

#endif
