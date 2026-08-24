/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_BF_OUU_UPDATE_HPP
#define HDSA_BF_OUU_UPDATE_HPP

#include "HDSA_BF_OUU_Sol_Op_Interface.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface.hpp"
#include "HDSA_Hessian_Inversion.hpp"

namespace HDSA
{

  template <class RealT>
  class BF_OUU_Update
  {

  private:
    HDSA::Ptr<HDSA::BF_OUU_Sol_Op_Interface<RealT>> ouu_sol_op_interface_;
    HDSA::Ptr<HDSA::MD_OUU_Opt_Prob_Interface<RealT>> ouu_opt_prob_interface_;
    HDSA::Ptr<HDSA::Hessian_Inversion<RealT>> hess_invert_;

  public:
    BF_OUU_Update(const HDSA::Ptr<HDSA::BF_OUU_Sol_Op_Interface<RealT>> &ouu_sol_op_interface, const HDSA::Ptr<HDSA::MD_OUU_Opt_Prob_Interface<RealT>> &ouu_opt_prob_interface, int verbosity = 0, RealT hessian_tol = 1.e-6, std::string hessian_solver = "CG") : ouu_sol_op_interface_(ouu_sol_op_interface), ouu_opt_prob_interface_(ouu_opt_prob_interface)
    {
      hess_invert_ = HDSA::makePtr<BF_OUU_Hessian_Inversion<RealT>>(ouu_opt_prob_interface, verbosity, hessian_tol, hessian_solver);
    }

    ~BF_OUU_Update(void)
    {
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Update(const HDSA::Vector<RealT> &u_lofi, const HDSA::Vector<RealT> &z_lofi) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> B = z_lofi.Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> z_update = z_lofi.Clone();

      HDSA::Ptr<HDSA::Vector<RealT>> u_hifi = u_lofi.Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> discrep = u_lofi.Clone();

      const HDSA::Ensemble_Vector<RealT> u_lofi_ens = dynamic_cast<const HDSA::Ensemble_Vector<RealT> &>(u_lofi);
      HDSA::Ensemble_Vector<RealT> u_hifi_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(*u_hifi);
      HDSA::Ensemble_Vector<RealT> discrep_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(*discrep);

      int M = u_lofi_ens.Number_of_Vectors();
      for(int s = 0; s < M; s++)
      {
        ouu_sol_op_interface_->State_Solve_Per_Sample(*u_hifi_ens[s], z_lofi, s);
        discrep_ens[s]->Set(*u_hifi_ens[s]);
        discrep_ens[s]->Scaled_Plus(-1.0,*u_lofi_ens[s]);
      }

      HDSA::Ptr<HDSA::Vector<RealT>> J_grad_u = u_lofi.Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> Hess_discrep = u_lofi.Clone();

      ouu_opt_prob_interface_->Misfit_Gradient(*J_grad_u, u_lofi, z_lofi);
      ouu_opt_prob_interface_->Apply_Misfit_Hessian(*Hess_discrep, *discrep, u_lofi, z_lofi);

      HDSA::Ensemble_Vector<RealT> J_grad_u_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(*J_grad_u);
      HDSA::Ensemble_Vector<RealT> Hess_discrep_ens = dynamic_cast<HDSA::Ensemble_Vector<RealT> &>(*Hess_discrep);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z_lofi.Clone();
      for(int s = 0; s < M; s++)
      {
        z_tmp->Zeros();
        ouu_sol_op_interface_->Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(*z_tmp, *J_grad_u_ens[s], z_lofi, s);
        B->Plus(*z_tmp);

        z_tmp->Zeros();
        ouu_opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(*z_tmp, *J_grad_u_ens[s], z_lofi, s);
        B->Scaled_Plus(-1.0, *z_tmp);

        z_tmp->Zeros();
        ouu_opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(*z_tmp, *Hess_discrep_ens[s], z_lofi, s);
        B->Plus(*z_tmp);
      }

      B->Scale(-1.0);
      hess_invert_->Apply_RS_Hessian_Inverse(*z_update, *B, z_lofi);
      z_update->Plus(z_lofi);
      return z_update;
    }

    template <class ScalarType>
    class BF_OUU_Hessian_Inversion : public HDSA::Hessian_Inversion<ScalarType>
    {
    private:
      HDSA::Ptr<HDSA::MD_OUU_Opt_Prob_Interface<ScalarType>> ouu_opt_prob_interface_;

    public:
      BF_OUU_Hessian_Inversion(const HDSA::Ptr<HDSA::MD_OUU_Opt_Prob_Interface<ScalarType>> &ouu_opt_prob_interface, int verbosity, ScalarType tol, std::string solver) : HDSA::Hessian_Inversion<ScalarType>(verbosity, tol, solver), ouu_opt_prob_interface_(ouu_opt_prob_interface)
      {
      }

      ~BF_OUU_Hessian_Inversion()
      {
      }

      void Apply_RS_Hessian(HDSA::Vector<ScalarType> &z_out, const HDSA::Vector<ScalarType> &z_in, const HDSA::Vector<ScalarType> &z) const
      {
        ouu_opt_prob_interface_->Apply_RS_Hessian(z_out, z_in, z);
      }
    };
  };

}

#endif
