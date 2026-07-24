/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_QUASI_NEWTON_PRECONDITIONER_HPP
#define HDSA_MD_QUASI_NEWTON_PRECONDITIONER_HPP

#include "HDSA_PC_Quasi_Newton_Preconditioner.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Quasi_Newton_Preconditioner : public PC_Quasi_Newton_Preconditioner<RealT>
  {

  private:
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;

  public:
    MD_Quasi_Newton_Preconditioner(const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis)
        : PC_Quasi_Newton_Preconditioner<RealT>(), hessian_analysis_(hessian_analysis)
    {
    }

    virtual ~MD_Quasi_Newton_Preconditioner()
    {
    }

  protected:

    void Apply_Initial_Inverse_Hessian_Approximation(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta_in) const override {
        beta_out.Set(beta_in);

        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = hessian_analysis_->Get_Evals();

        if (evals == HDSA::nullPtr) {
            return;
        }

        int r = std::min(beta_out.Dimension(), evals->Number_of_Rows());

        constexpr RealT eps = static_cast<RealT>(1e-14);

        for (int i = 0; i < r; ++i)
        {
            RealT lambda = (*evals)(i,0);
            if (std::abs(lambda) > eps) {
                beta_out.Set_Entry(i, beta_out.Get_Entry(i)/lambda);
            }
        }
    }
  };

}

#endif