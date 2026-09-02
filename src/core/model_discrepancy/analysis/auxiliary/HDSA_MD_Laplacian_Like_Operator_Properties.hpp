/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_LAPLACIAN_LIKE_OPERATOR_PROPERTIES_HPP
#define HDSA_MD_LAPLACIAN_LIKE_OPERATOR_PROPERTIES_HPP

#include "HDSA_MD_Scaled_u_Prior_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Laplacian_Like_Operator_Properties
  {

  private:
  public:
    MD_Laplacian_Like_Operator_Properties()
    {
    }

    virtual ~MD_Laplacian_Like_Operator_Properties()
    {
    }

    RealT Get_Rectangular_Domain_Squared_Inv_Operator_Trace(const RealT &beta, const std::vector<std::vector<RealT>> &spatial_bounds, const int &n_u)
    {
      std::vector<RealT> evals;
      int d = spatial_bounds.size();
      if (d == 1)
      {
        RealT Lx = spatial_bounds[0][1] - spatial_bounds[0][0];
        int n = n_u - 1;
        evals.resize(n + 1);
        for (int i = 0; i < n + 1; i++)
        {
          evals[i] = 1.0 / (1.0 + beta * std::pow(M_PI / Lx, 2.0) * std::pow(static_cast<RealT>(i), 2.0));
        }
      }
      else if (d == 2)
      {
        RealT Lx = spatial_bounds[0][1] - spatial_bounds[0][0];
        RealT Ly = spatial_bounds[1][1] - spatial_bounds[1][0];
        int n = std::round(std::sqrt(n_u)) - 1;
        evals.resize((n + 1) * (n + 1));
        int count = 0;
        for (int i = 0; i < n + 1; i++)
        {
          for (int j = 0; j < n + 1; j++)
          {
            evals[count] = 1.0 / (1.0 + beta * (std::pow(M_PI / Lx, 2.0) * std::pow(static_cast<RealT>(i), 2.0) + std::pow(M_PI / Ly, 2.0) * std::pow(static_cast<RealT>(j), 2.0)));
            count += 1;
          }
        }
      }
      else if (d == 3)
      {
        RealT Lx = spatial_bounds[0][1] - spatial_bounds[0][0];
        RealT Ly = spatial_bounds[1][1] - spatial_bounds[1][0];
        RealT Lz = spatial_bounds[2][1] - spatial_bounds[2][0];
        int n = std::round(std::pow(n_u, 1.0 / 3.0)) - 1;
        evals.resize((n + 1) * (n + 1) * (n + 1));
        int count = 0;
        for (int i = 0; i < n + 1; i++)
        {
          for (int j = 0; j < n + 1; j++)
          {
            for (int k = 0; k < n + 1; k++)
            {
              evals[count] = 1.0 / (1.0 + beta * (std::pow(M_PI / Lx, 2.0) * std::pow(static_cast<RealT>(i), 2.0) + std::pow(M_PI / Ly, 2.0) * std::pow(static_cast<RealT>(j), 2.0)) + std::pow(M_PI / Lz, 2.0) * std::pow(static_cast<RealT>(k), 2.0));
              count += 1;
            }
          }
        }
      }
      else
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA::MD_Laplacian_Like_Operator_Properties: The code only supports spatial Dimensions s=1,2,3" << std::endl);
      }

      RealT val = 0.0;
      for(long unsigned int k = 0; k < evals.size(); k++)
      {
        val += std::pow(evals[k],2.0);
      }
      return val;
    }

    RealT Randomized_Inv_Operator_Trace_Estimation(const HDSA::MD_Scaled_u_Prior_Interface<RealT> &u_prior_interface, HDSA::Ptr<HDSA::Vector<RealT>> &u_vec, const int &num_samples)
    {
      HDSA::Ptr<HDSA::MultiVector<RealT>> samples = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_samples, *u_vec);
      u_prior_interface.Sample_with_Covariance_W_u_Acute_Inverse(*samples);
      RealT val = 0.0;
      for (int s = 0; s < num_samples; s++)
      {
        u_vec->Zeros();
        u_prior_interface.Apply_M_u(*u_vec, *(*samples)[s]);
        val += (*samples)[s]->Dot(*u_vec);
      }
      val = val / static_cast<RealT>(num_samples);
      return val;
    }
  };

}

#endif
