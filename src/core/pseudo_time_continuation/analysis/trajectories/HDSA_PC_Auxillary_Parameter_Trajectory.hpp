/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_AUXILLARY_PARAMETER_TRAJECTORY_HPP
#define HDSA_PC_AUXILLARY_PARAMETER_TRAJECTORY_HPP

namespace HDSA
{

  template <class RealT>
  class PC_Auxillary_Parameter_Trajectory
  {

  private:
    int N_;

  public:
    PC_Auxillary_Parameter_Trajectory(int N): N_(N)
    {
    }

    virtual ~PC_Auxillary_Parameter_Trajectory()
    {
    }
    
    int Get_Number_of_Timesteps(void) const 
    {
      return N_;
    }

  };

}

#endif
