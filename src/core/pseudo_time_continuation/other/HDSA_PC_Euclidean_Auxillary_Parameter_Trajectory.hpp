/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_EUCLIDEAN_AUXILLARY_PARAMETER_TRAJECTORY_HPP
#define HDSA_PC_EUCLIDEAN_AUXILLARY_PARAMETER_TRAJECTORY_HPP

#include "HDSA_PC_Auxillary_Parameter_Trajectory.hpp"
#include "HDSA_Vector.hpp"

namespace HDSA
{

  template <class RealT>
  class PC_Euclidean_Auxillary_Parameter_Trajectory : public PC_Auxillary_Parameter_Trajectory<RealT>
  {

  private:
    HDSA::Ptr<HDSA::Vector<RealT>> theta_bar_;
    HDSA::Ptr<HDSA::Vector<RealT>> theta_star_;
    HDSA::Ptr<HDSA::Vector<RealT>> dtheta_;

  public:
    PC_Euclidean_Auxillary_Parameter_Trajectory(int N, HDSA::Ptr<HDSA::Vector<RealT>> & theta_bar, HDSA::Ptr<HDSA::Vector<RealT>> & theta_star)
        : PC_Auxillary_Parameter_Trajectory<RealT>(N), theta_bar_(theta_bar), theta_star_(theta_star)
    {
      dtheta_ = theta_bar->Clone();
      dtheta_->Set(*theta_star_);
      dtheta_->Scaled_Plus(-1.0, *theta_bar);
    }

    virtual ~PC_Euclidean_Auxillary_Parameter_Trajectory()
    {
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Get_dtheta(void) const
    {
      return dtheta_;
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Get_theta_n(RealT n) const
    {
      int N = this->Get_Number_of_Timesteps();
      if (n < 0 || n > N)
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory: Get_theta_n: The time index n is out of range" << std::endl);
      }
      HDSA::Ptr<HDSA::Vector<RealT>> theta_n = theta_bar_->Clone();
      theta_n->Set(*theta_bar_);
      RealT alpha = n/static_cast<RealT>(N);
      theta_n->Scaled_Plus(alpha,*dtheta_);
      return theta_n;
    }

  };

}

#endif
