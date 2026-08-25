/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_NUMERIC_LAPLACIAN_Z_PRIOR_INTERFACE_HPP
#define HDSA_MD_NUMERIC_LAPLACIAN_Z_PRIOR_INTERFACE_HPP

#include "HDSA_MD_Elliptic_z_Prior_Interface.hpp"
#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_Sparse_Matrix_Solver.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface.hpp"
#include "HDSA_MD_Determine_z_Hyperparameters_Decl.hpp"
#include "HDSA_Matrix_Sqrt.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Numeric_Laplacian_z_Prior_Interface : public HDSA::MD_Elliptic_z_Prior_Interface<RealT>
  {

  private:
    const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_;
    const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_;
    const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
    const HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> z_hyperparam_interface_;
    const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
    bool use_direct_solvers_;
    int verbosity_;
    bool use_incomplete_prec_;
    std::ostream &out_stream_;
    HDSA::Ptr<HDSA::MD_Determine_z_Hyperparameters<RealT>> determine_z_hyperparams_;
    RealT beta_z_;
    HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> E_z_;
    HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> E_z_solver_;
    HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> M_z_solver_;
    HDSA::Ptr<HDSA::Sparse_Matrix_Sqrt<RealT>> M_z_sqrt_;
    HDSA::Ptr<HDSA::Timer<RealT>> timer_;

  public:
    void Apply_E_z_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      if (verbosity_ > 3)
      {
        timer_->Start_Timer();
      }
      std::string output_message = E_z_solver_->Apply_A_Inverse(z_out, z_in);
      if (verbosity_ > 3)
      {
        RealT elapsed_time = timer_->End_Timer();
        std::cout << output_message << " in " << elapsed_time << " seconds." << std::endl;
      }
    }

    void Apply_E_z_Inverse_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      if (verbosity_ > 3)
      {
        timer_->Start_Timer();
      }
      std::string output_message = E_z_solver_->Apply_A_Inverse(z_out, z_in);
      if (verbosity_ > 3)
      {
        RealT elapsed_time = timer_->End_Timer();
        std::cout << output_message << " in " << elapsed_time << " seconds." << std::endl;
      }
    }

    void Apply_M_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      M_->Apply(z_out, z_in);
    }

    virtual void Sample_with_Covariance_W_z_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      samples.Zeros();
      for (int k = 0; k < samples.Number_of_Vectors(); k++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> omega = samples[k]->Clone();
        omega->Randomize_Standard_Normal();
        HDSA::Ptr<HDSA::Vector<RealT>> vec = samples[k]->Clone();
        std::string output_message = M_z_sqrt_->Matrix_Sqrt_Apply(*vec, *omega);
        if (verbosity_ > 3)
        {
          RealT elapsed_time = timer_->End_Timer();
          std::cout << output_message << " in " << elapsed_time << " seconds." << std::endl;
        }

        Apply_E_z_Inverse(*samples[k], *vec);
      }
    }

    virtual void Apply_E_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      E_z_->Apply(z_out, z_in);
    }

    virtual void Apply_E_z_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      E_z_->Apply(z_out, z_in);
    }

    virtual void Apply_M_z_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      if (verbosity_ > 3)
      {
        timer_->Start_Timer();
      }
      std::string output_message = M_z_solver_->Apply_A_Inverse(z_out, z_in);
      if (verbosity_ > 3)
      {
        RealT elapsed_time = timer_->End_Timer();
        std::cout << output_message << " in " << elapsed_time << " seconds." << std::endl;
      }
    }

    MD_Numeric_Laplacian_z_Prior_Interface(const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> &S, const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> &M, const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, const HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> &z_hyperparam_interface, const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface, const bool use_direct_solvers = true, const int verbosity = 0, const bool use_incomplete_prec = false, std::ostream &out_stream = std::cout) : HDSA::MD_Elliptic_z_Prior_Interface<RealT>(z_hyperparam_interface->Get_alpha_z()), S_(S), M_(M), data_interface_(data_interface), z_hyperparam_interface_(z_hyperparam_interface), u_prior_interface_(u_prior_interface), use_direct_solvers_(use_direct_solvers), verbosity_(verbosity), use_incomplete_prec_(use_incomplete_prec), out_stream_(out_stream)
    {
      timer_ = HDSA::makePtr<HDSA::Timer<RealT>>(out_stream_);
      E_z_ = M_->Clone();
      determine_z_hyperparams_ = HDSA::makePtr<HDSA::MD_Determine_z_Hyperparameters<RealT>>(data_interface_, z_hyperparam_interface_, u_prior_interface_);

      M_->Set_Symmetric();
      std::string A_solver_message = "M_z_Inverse";
      M_z_solver_ = M_->Get_Sparse_Matrix_Solver(use_direct_solvers_, verbosity_, out_stream_, A_solver_message);
      std::string A_sqrt_solver_message = "M_z_Sqrt";
      M_z_sqrt_ = HDSA::makePtr<HDSA::Sparse_Matrix_Sqrt<RealT>>(M_, A_sqrt_solver_message);
      if (use_incomplete_prec_)
      {
        if (verbosity_ > 3)
        {
          timer_->Start_Timer();
        }
        HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = M_->Get_Incomplete_Chol_Factor();
        M_z_solver_->Set_Incomplete_Factor(L);
        M_z_sqrt_->Set_Incomplete_Factor(L);
        if (verbosity_ > 3)
        {
          RealT elaped_time = timer_->End_Timer();
          out_stream_ << "M_z incomplete factorization took " << elaped_time << " seconds." << std::endl;
        }
      }

      if (z_hyperparam_interface_->Get_beta_z() == 0.0)
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA::MD_Numeric_Laplacian_z_Prior_Interface: The value of beta_z must be specificed" << std::endl);
      }
      Set_beta_z(z_hyperparam_interface_->Get_beta_z());

      if (z_hyperparam_interface_->Get_alpha_z() == 0.0)
      {
        determine_z_hyperparams_->Determine_alpha_z(this);
      }
      this->Set_alpha_z(z_hyperparam_interface_->Get_alpha_z());
    }

    virtual ~MD_Numeric_Laplacian_z_Prior_Interface()
    {
    }

    void Set_beta_z(RealT beta_z_new)
    {
      E_z_->Set(*M_);
      E_z_->Scaled_Plus(beta_z_new, *S_);
      beta_z_ = beta_z_new;
      E_z_->Set_Symmetric();
      std::string A_solver_message = "E_z_Inverse";
      E_z_solver_ = E_z_->Get_Sparse_Matrix_Solver(use_direct_solvers_, verbosity_, out_stream_, A_solver_message);
      if (use_incomplete_prec_)
      {
        if (verbosity_ > 3)
        {
          timer_->Start_Timer();
        }
        HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = E_z_->Get_Incomplete_Chol_Factor();
        E_z_solver_->Set_Incomplete_Factor(L);
        if (verbosity_ > 3)
        {
          RealT elaped_time = timer_->End_Timer();
          out_stream_ << "E_z incomplete factorization took " << elaped_time << " seconds." << std::endl;
        }
      }
    }
  };

}

#include "HDSA_MD_Determine_z_Hyperparameters_Def.hpp"

#endif
