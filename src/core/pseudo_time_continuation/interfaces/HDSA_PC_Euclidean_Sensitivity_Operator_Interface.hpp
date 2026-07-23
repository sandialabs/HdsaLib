/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_EUCLIDEAN_SENSITIVITY_OPERATOR_INTERFACE_HPP
#define HDSA_PC_EUCLIDEAN_SENSITIVITY_OPERATOR_INTERFACE_HPP

#include "HDSA_PC_Euclidean_Auxillary_Parameter_Trajectory.hpp"

namespace HDSA
{

  template <class RealT>
  class PC_Euclidean_Sensitivity_Operator_Interface : public PC_Sensitivity_Operator_Interface<RealT>
  {

  private:
  public:
    PC_Euclidean_Sensitivity_Operator_Interface()
    {
    }

    virtual ~PC_Euclidean_Sensitivity_Operator_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Euclidean_Gradient(HDSA::Vector<RealT> &grad, const HDSA::Vector<RealT> &z, const HDSA::Vector<RealT> &theta) const = 0;

    virtual void Euclidean_Apply_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, const HDSA::Vector<RealT> &theta) const = 0;

    virtual void Euclidean_Apply_B(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &theta_in, const HDSA::Vector<RealT> &z, const HDSA::Vector<RealT> &theta) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of base class pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Gradient(HDSA::Vector<RealT> &grad, const HDSA::Vector<RealT> &z, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const
    {
      const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *euclidean_theta_traj = dynamic_cast<const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *>(&theta_traj);
      HDSA::Ptr<HDSA::Vector<RealT>> theta = euclidean_theta_traj->Get_theta_n(time_index);

      Euclidean_Gradient(grad, z, *theta);
    }

    void Apply_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const
    {
      const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *euclidean_theta_traj = dynamic_cast<const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *>(&theta_traj);
      HDSA::Ptr<HDSA::Vector<RealT>> theta = euclidean_theta_traj->Get_theta_n(time_index);

      Euclidean_Apply_Hessian(z_out, z_in, z, *theta);
    }

    void Apply_B(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const
    {
      const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *euclidean_theta_traj = dynamic_cast<const HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT> *>(&theta_traj);
      HDSA::Ptr<HDSA::Vector<RealT>> theta = euclidean_theta_traj->Get_theta_n(time_index);
      HDSA::Ptr<HDSA::Vector<RealT>> theta_in = euclidean_theta_traj->Get_dtheta();

      Euclidean_Apply_B(z_out, *theta_in, z, *theta);
    }
  };

}

#endif
