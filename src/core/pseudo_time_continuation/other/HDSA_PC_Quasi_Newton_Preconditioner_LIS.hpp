/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_QUASI_NEWTON_PRECONDITIONER_LIS_HPP
#define HDSA_PC_QUASI_NEWTON_PRECONDITIONER_LIS_HPP

#include "HDSA_PC_Quasi_Newton_Preconditioner.hpp"
#include "HDSA_PC_LIS_Interface.hpp"
#include "HDSA_Randomized_GEVP.hpp"

namespace HDSA
{

  template <class RealT>
  class PC_Quasi_Newton_Preconditioner_LIS : public HDSA::PC_Quasi_Newton_Preconditioner<RealT>
  {

  private:
    HDSA::Ptr<HDSA::PC_LIS_Interface<RealT>> lis_interface_;
    HDSA::Ptr<HDSA::Vector<RealT>> z_bar_;
    HDSA::Ptr<HDSA::Vector<RealT>> theta_bar_;
    int rank_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> evecs_;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals_;

  public:
    PC_Quasi_Newton_Preconditioner_LIS(const HDSA::Ptr<HDSA::Vector<RealT>> &z_bar, const HDSA::Ptr<HDSA::Vector<RealT>> &theta_bar,
                                       const HDSA::Ptr<HDSA::PC_LIS_Interface<RealT>> &lis_interface, bool use_block_update = true, RealT tau = 1.e-6, int max_storage = 10) : HDSA::PC_Quasi_Newton_Preconditioner<RealT>(use_block_update, tau, max_storage), lis_interface_(lis_interface), z_bar_(z_bar), theta_bar_(theta_bar)
    {
      rank_ = 0;
    }

    virtual ~PC_Quasi_Newton_Preconditioner_LIS()
    {
    }

    void Compute_Hessian_GEVP(const HDSA::Ptr<HDSA::Vector<RealT>> &z, const HDSA::Ptr<HDSA::Vector<RealT>> &theta, const int &num_evals, const int &oversampling)
    {
      rank_ = num_evals;
      HDSA::Ptr<HDSA::Randomized_GEVP<RealT>> hessian_gevp = HDSA::makePtr<PC_LIS_Hessian_GEVP<RealT>>(lis_interface_, z_bar_, theta_bar_);
      evecs_ = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_evals, *z_bar_);
      evals_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(num_evals, 1);
      hessian_gevp->Compute_GEVP(*evecs_, *evals_, num_evals, oversampling);
      std::ofstream fout;
      std::string name = "hessian_evals.txt";
      fout.open(name);
      for (int i = 0; i < num_evals; i++)
      {
        fout << std::setprecision(16) << (*evals_)(i, 0) << "  ";
      }
      fout.close();
      HDSA::PC_Quasi_Newton_Preconditioner<RealT>::total_vecs_stored_ += num_evals;
      HDSA::PC_Quasi_Newton_Preconditioner<RealT>::num_Hvecs_ += 2 * (num_evals + oversampling);
    }

  protected:
    // Overload this function if a better initialization is available
    void Apply_Initial_Inverse_Hessian_Approximation(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      lis_interface_->Apply_Prior_Covariance(z_out, z_in);
      if (rank_ > 0)
      {
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp = evecs_->MatVec(z_in);
        for (int k = 0; k < rank_; k++)
        {
          RealT val = (*tmp)(k, 0) * (*evals_)(k, 0) / (1.0 + (*evals_)(k, 0));
          z_out.Scaled_Plus(-val, *(*evecs_)[k]);
        }
      }
    }

    template <class ScalarType>
    class PC_LIS_Hessian_GEVP : public HDSA::Randomized_GEVP<ScalarType>
    {

    private:
      HDSA::Ptr<HDSA::Vector<ScalarType>> z_bar_;
      HDSA::Ptr<HDSA::Vector<ScalarType>> theta_bar_;
      HDSA::Ptr<HDSA::PC_LIS_Interface<ScalarType>> lis_interface_;

    public:
      PC_LIS_Hessian_GEVP(const HDSA::Ptr<HDSA::PC_LIS_Interface<ScalarType>> &lis_interface, const HDSA::Ptr<HDSA::Vector<ScalarType>> &z_bar, const HDSA::Ptr<HDSA::Vector<ScalarType>> &theta_bar)
          : HDSA::Randomized_GEVP<ScalarType>()
      {
        z_bar_ = z_bar;
        theta_bar_ = theta_bar;
        lis_interface_ = lis_interface;
      }

      virtual ~PC_LIS_Hessian_GEVP()
      {
      }

      void Apply_Operator(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        lis_interface_->Apply_Misfit_Hessian(vec_out, vec_in, *z_bar_, *theta_bar_);
      }

      void Apply_Weighting_Operator(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        lis_interface_->Apply_Prior_Precision(vec_out, vec_in);
      }

      void Apply_Weighting_Operator_Inverse(HDSA::Vector<ScalarType> &vec_out, const HDSA::Vector<ScalarType> &vec_in) const
      {
        lis_interface_->Apply_Prior_Covariance(vec_out, vec_in);
      }

      void Generate_Random_Samples(HDSA::MultiVector<RealT> &samples) const
      {
        lis_interface_->Generate_Prior_Samples(samples);
      }
    };
  };

}

#endif
