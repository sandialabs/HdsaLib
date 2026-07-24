/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DISCREPANCY_PARAMETER_TRAJECTORY_HPP
#define HDSA_MD_DISCREPANCY_PARAMETER_TRAJECTORY_HPP

#include "HDSA_PC_Auxillary_Parameter_Trajectory.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Discrepancy_Parameter_Trajectory : public PC_Auxillary_Parameter_Trajectory<RealT>
  {

  private:
    int sample_idx_;

  public:
    MD_Discrepancy_Parameter_Trajectory(int num_continuation_steps, int sample_index)
        : PC_Auxillary_Parameter_Trajectory<RealT>(num_continuation_steps), sample_idx_(sample_index)
    {
    }

    virtual ~MD_Discrepancy_Parameter_Trajectory()
    {
    }

    RealT Get_Time(RealT time_index) const {
      return time_index / static_cast<RealT>(this->Get_Number_of_Timesteps());
    }

    int Get_Sample_Index(void) const
    {
      return sample_idx_;
    }

    int Number_of_Timesteps() const
    {
        return this->Get_Number_of_Timesteps();
    }
  };

}

#endif