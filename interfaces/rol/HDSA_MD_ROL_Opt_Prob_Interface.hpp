/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_ROL_OPT_PROB_INTERFACE_HPP
#define HDSA_MD_ROL_OPT_PROB_INTERFACE_HPP

#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "HDSA_MD_Opt_Prob_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_ROL_Opt_Prob_Interface : public HDSA::MD_Opt_Prob_Interface<RealT>
  {

  private:
    ROL::Ptr<ROL::Objective_SimOpt<RealT>> obj_simopt_;
    ROL::Ptr<ROL::Constraint_SimOpt<RealT>> con_simopt_;
    ROL::Ptr<ROL::Reduced_Objective_SimOpt<RealT>> red_obj_;

  public:
    MD_ROL_Opt_Prob_Interface(ROL::Ptr<ROL::Objective_SimOpt<RealT>> &obj_simopt, ROL::Ptr<ROL::Constraint_SimOpt<RealT>> &con_simopt,
                              ROL::Ptr<ROL::Vector<RealT>> &u, ROL::Ptr<ROL::Vector<RealT>> &z) : obj_simopt_(obj_simopt), con_simopt_(con_simopt)
    {
      ROL::Ptr<ROL::Vector<RealT>> p = u->clone();
      red_obj_ = ROL::makePtr<ROL::Reduced_Objective_SimOpt<RealT>>(obj_simopt, con_simopt, u, z, p);
    }

    virtual ~MD_ROL_Opt_Prob_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions (from the base class) that are implemented using the SimOpt interface
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const
    {
      HDSA::ROL_Vector<RealT> &z_out_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(z_out);
      const HDSA::ROL_Vector<RealT> &u_in_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(u_in);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = 1.e-8;
      ROL::Ptr<ROL::Vector<RealT>> u_tmp = u_in_rol.rol_vec->clone();
      ROL::Ptr<ROL::Vector<RealT>> u_rol_vec = u_in_rol.rol_vec->clone();
      ROL::Ptr<ROL::Vector<RealT>> c_rol_vec = u_in_rol.rol_vec->clone();
      con_simopt_->solve(*c_rol_vec, *u_rol_vec, *z_rol.rol_vec, tol);
      con_simopt_->applyInverseAdjointJacobian_1(*u_tmp, *u_in_rol.rol_vec, *u_rol_vec, *z_rol.rol_vec, tol);
      con_simopt_->applyAdjointJacobian_2(*z_out_rol.rol_vec, *u_tmp, *u_rol_vec, *z_rol.rol_vec, tol);
      z_out.Scale(-1.0);
    }

    void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
    {
      HDSA::ROL_Vector<RealT> &z_out_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(z_out);
      const HDSA::ROL_Vector<RealT> &z_in_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z_in);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = 1.e-8;
      red_obj_->update(*z_rol.rol_vec, ROL::UpdateType::Temp);
      red_obj_->hessVec(*z_out_rol.rol_vec, *z_in_rol.rol_vec, *z_rol.rol_vec, tol);
    }

    void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
    {
      HDSA::ROL_Vector<RealT> &u_grad_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(u_grad);
      const HDSA::ROL_Vector<RealT> &u_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(u);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = 1.e-8;
      obj_simopt_->update(*u_rol.rol_vec, *z_rol.rol_vec, ROL::UpdateType::Temp);
      obj_simopt_->gradient_1(*u_grad_rol.rol_vec, *u_rol.rol_vec, *z_rol.rol_vec, tol);
    }

    void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
    {
      HDSA::ROL_Vector<RealT> &u_out_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(u_out);
      const HDSA::ROL_Vector<RealT> &u_in_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(u_in);
      const HDSA::ROL_Vector<RealT> &u_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(u);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = 1.e-8;
      obj_simopt_->update(*u_rol.rol_vec, *z_rol.rol_vec, ROL::UpdateType::Temp);
      obj_simopt_->hessVec_11(*u_out_rol.rol_vec, *u_in_rol.rol_vec, *u_rol.rol_vec, *z_rol.rol_vec, tol);
    }

    void Apply_Solution_Operator_z_Jacobian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const {
      HDSA::ROL_Vector<RealT> &u_out_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(u_out);
      const HDSA::ROL_Vector<RealT> &z_in_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z_in);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = static_cast<RealT>(1.e-8);
      ROL::Ptr<ROL::Vector<RealT>> u_rol_vec = u_out_rol.rol_vec->clone();
      ROL::Ptr<ROL::Vector<RealT>> c_rol_vec = u_out_rol.rol_vec->clone();
      con_simopt_->solve(*c_rol_vec, *u_rol_vec, *z_rol.rol_vec, tol);
      ROL::Ptr<ROL::Vector<RealT>> c_z_zin = u_out_rol.rol_vec->clone();
      con_simopt_->applyJacobian_2(*c_z_zin, *z_in_rol.rol_vec, *u_rol_vec, *z_rol.rol_vec, tol);
      con_simopt_->applyInverseJacobian_1(*u_out_rol.rol_vec, *c_z_zin, *u_rol_vec, *z_rol.rol_vec, tol);
      u_out.Scale(static_cast<RealT>(-1));
    }

    void State_Solve(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z) const {
      HDSA::ROL_Vector<RealT> &u_out_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(u_out);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = static_cast<RealT>(1.e-8);
      ROL::Ptr<ROL::Vector<RealT>> c_rol_vec = u_out_rol.rol_vec->clone();
      con_simopt_->solve(*c_rol_vec, *u_out_rol.rol_vec, *z_rol.rol_vec, tol);
    }

    RealT Objective_Function(HDSA::Vector<RealT> &grad_u, HDSA::Vector<RealT> &grad_z, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const {
      HDSA::ROL_Vector<RealT> &grad_u_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(grad_u);
      HDSA::ROL_Vector<RealT> &grad_z_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(grad_z);
      const HDSA::ROL_Vector<RealT> &u_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(u);
      const HDSA::ROL_Vector<RealT> &z_rol = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(z);
      RealT tol = static_cast<RealT>(1.e-8);
      obj_simopt_->update(*u_rol.rol_vec, *z_rol.rol_vec, ROL::UpdateType::Temp);
      const RealT value = obj_simopt_->value(*u_rol.rol_vec, *z_rol.rol_vec, tol);
      obj_simopt_->gradient_1(*grad_u_rol.rol_vec, *u_rol.rol_vec, *z_rol.rol_vec, tol);
      obj_simopt_->gradient_2(*grad_z_rol.rol_vec, *u_rol.rol_vec, *z_rol.rol_vec, tol);
      return value;
    }
  };

}

#endif
