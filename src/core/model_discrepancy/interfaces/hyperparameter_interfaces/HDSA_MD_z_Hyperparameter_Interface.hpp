/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_Z_HYPERPARAMETER_INTERFACE_HPP
#define HDSA_MD_Z_HYPERPARAMETER_INTERFACE_HPP

#include "HDSA_Random_Number_Generator.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_z_Hyperparameter_Interface
  {

  private:
    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
    std::string z_type_;
    int num_state_solves_;
    RealT discrepancy_percent_z_variation_;

    RealT alpha_z_;
    RealT beta_z_;
    RealT beta_t_;

  public:
    virtual std::vector<std::vector<RealT>> Spatial_Domain_Bounds(void) const
    {
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_z_Hyperparameter_Interface::Spatial_Domain_Bounds must be implemented for hyperparameter algorithm-based initialization" << std::endl);
      std::vector<std::vector<RealT>> vec; // vec.size() = spatial Dimension, e.g. 1,2, or 3, [ vec[i][0],vec[i][1] ] is an interval bounding the ith spatial coordinate
      return vec;
    }

    virtual void State_Solve(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
    {
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_z_Hyperparameter_Interface: State_Solve must be implemented to estimate alpha_z using low-fidelity solves" << std::endl);
    }

    MD_z_Hyperparameter_Interface(const std::string &z_type, const int &num_state_solves = 0) : random_number_generator_(HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>()), z_type_(z_type), num_state_solves_(num_state_solves)
    {
      Auxillary_Constructor();
    }

    MD_z_Hyperparameter_Interface(int seed, const std::string &z_type, const int &num_state_solves = 0) : random_number_generator_(HDSA::Random_Number_Generator<RealT>(seed)), z_type_(z_type), num_state_solves_(num_state_solves)
    {
      Auxillary_Constructor();
    }

    MD_z_Hyperparameter_Interface(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, const std::string &z_type, const int &num_state_solves = 0) : random_number_generator_(random_number_generator), z_type_(z_type), num_state_solves_(num_state_solves)
    {
      Auxillary_Constructor();
    }

    void Auxillary_Constructor()
    {
      if (!(z_type_ == "spatial field" || z_type_ == "transient vector" || z_type_ == "vector"))
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA::MD_z_Hyperparameter_Interface: The input z_type should be either 'spatial field' 'transient vector' or 'vector'" << std::endl);
      }

      discrepancy_percent_z_variation_ = 1.0;

      alpha_z_ = 0.0;
      beta_z_ = 0.0;
      beta_t_ = 0.0;
    }

    virtual ~MD_z_Hyperparameter_Interface()
    {
    }

    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> Get_Random_Number_Generator(void) const
    {
      return random_number_generator_;
    }

    std::string Get_z_type(void) const
    {
      return z_type_;
    }

    void Set_Discrepancy_Percent_z_Variation(RealT val)
    {
      discrepancy_percent_z_variation_ = val;
    }

    RealT Get_Discrepancy_Percent_z_Variation(void) const
    {
      return discrepancy_percent_z_variation_;
    }

    int Get_num_state_solves(void) const
    {
      return num_state_solves_;
    }

    void Set_alpha_z(RealT alpha_z_new)
    {
      alpha_z_ = alpha_z_new;
    }

    RealT Get_alpha_z(void) const
    {
      return alpha_z_;
    }

    void Set_beta_z(RealT beta_z_new)
    {
      beta_z_ = beta_z_new;
    }

    RealT Get_beta_z(void) const
    {
      return beta_z_;
    }

    void Set_beta_t(RealT beta_t_new)
    {
      beta_t_ = beta_t_new;
    }

    RealT Get_beta_t(void) const
    {
      return beta_t_;
    }
  };

}

#endif
