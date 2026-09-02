/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_LUMPED_MASS_U_PRIOR_INTERFACE_HPP
#define HDSA_MD_LUMPED_MASS_U_PRIOR_INTERFACE_HPP

#include "HDSA_Linear_Algebra.hpp"
#include "HDSA_Linear_Operator.hpp"
#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_MD_Determine_u_Hyperparameters_Decl.hpp"
#include "HDSA_MD_Scaled_u_Prior_Interface.hpp"
#include "HDSA_MD_u_Hyperparameter_Interface.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_Sparse_Matrix_Solver.hpp"
#include "HDSA_Sparse_Matrix_Sqrt.hpp"
#include "HDSA_Timer.hpp"

namespace HDSA
{

template <class RealT> class MD_Lumped_Mass_u_Prior_Interface : public HDSA::MD_Scaled_u_Prior_Interface<RealT>
{

  private:
    const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_;
    const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_;
    const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
    const HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> u_hyperparam_interface_;
    bool use_direct_solvers_;
    int verbosity_;
    bool use_incomplete_prec_;
    std::ostream &out_stream_;
    HDSA::Ptr<HDSA::MD_Determine_u_Hyperparameters<RealT>> determine_u_hyperparams_;
    RealT beta_u_;
    HDSA::Ptr<HDSA::Vector<RealT>> M_lumped_;
    HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> E_u_;
    HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> E_u_solver_;
    HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> W_u_acute_;
    std::vector<RealT> scalars_;
    std::vector<HDSA::Ptr<HDSA::Sparse_Matrix<RealT>>> W_u_acute_plus_scalar_M_u_;
    std::vector<HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>>> W_u_acute_plus_scalar_M_u_solver_;
    std::vector<HDSA::Ptr<HDSA::Sparse_Matrix_Sqrt<RealT>>> W_u_acute_plus_scalar_M_u_sqrt_;
    HDSA::Ptr<HDSA::Timer<RealT>> timer_;

  public:
    void Apply_M_u(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const override
    {
        M_->Apply(u_out, u_in);
    }

    void Apply_W_u_Acute_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const override
    {
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = u_in.Clone();
        Apply_E_u_Inverse(*u_tmp, u_in);
        for (int k = 0; k < u_in.Dimension(); k++)
        {
            RealT val = u_tmp->Get_Entry(k) * M_lumped_->Get_Entry(k);
            u_tmp->Set_Entry(k, val);
        }
        Apply_E_u_Inverse(u_out, *u_tmp);
    }

    void Apply_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const override
    {
        int i = -1;
        for (long unsigned int k = 0; k < scalars_.size(); k++)
        {
            if (std::abs(scalars_[k] - scalar) < 1.e-12)
            {
                i = k;
                break;
            }
        }

        if (i < 0)
        {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                    "Error in HDSA::MD_Lumped_Mass_u_Prior_Interface::Apply_W_u_Acute_Plus_scalar_M_u_Inverse: scalar has not "
                                    "been set by Precompute_W_u_Plus_scalar_M_u_Data"
                                        << std::endl);
        }

        if (verbosity_ > 2)
        {
            timer_->Start_Timer();
        }
        std::string output_message = W_u_acute_plus_scalar_M_u_solver_[i]->Apply_A_Inverse(u_out, u_in);
        if (verbosity_ > 2)
        {
            RealT elapsed_time = timer_->End_Timer();
            out_stream_ << output_message << " in " << elapsed_time << " seconds." << std::endl;
        }
    }

    void Sample_with_Covariance_W_u_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const override
    {
        int num_samples = samples.Number_of_Vectors();
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = samples[0]->Clone();
        for (int s = 0; s < num_samples; s++)
        {
            u_tmp->Randomize_Standard_Normal();
            for (int k = 0; k < u_tmp->Dimension(); k++)
            {
                RealT val = u_tmp->Get_Entry(k) * std::sqrt(M_lumped_->Get_Entry(k));
                u_tmp->Set_Entry(k, val);
            }
            Apply_E_u_Inverse(*samples[s], *u_tmp);
        }
    }

    void Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const override
    {
        int i = -1;
        for (long unsigned int k = 0; k < scalars_.size(); k++)
        {
            if (std::abs(scalars_[k] - scalar) < 1.e-12)
            {
                i = k;
                break;
            }
        }

        if (i < 0)
        {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                    "Error in HDSA::MD_Lumped_Mass_u_Prior_Interface::Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse: "
                                    "scalar has not been set by Precompute_W_u_Plus_scalar_M_u_Data"
                                        << std::endl);
        }

        HDSA::Ptr<HDSA::Vector<RealT>> vec_rand = samples[0]->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> vec = samples[0]->Clone();
        for (int s = 0; s < samples.Number_of_Vectors(); s++)
        {
            vec_rand->Randomize_Standard_Normal();
            if (verbosity_ > 2)
            {
                timer_->Start_Timer();
            }
            std::string output_message = W_u_acute_plus_scalar_M_u_sqrt_[i]->Matrix_Sqrt_Apply(*vec, *vec_rand);
            if (verbosity_ > 2)
            {
                RealT elapsed_time = timer_->End_Timer();
                out_stream_ << output_message << " in " << elapsed_time << " seconds." << std::endl;
            }
            Apply_W_u_Acute_Plus_scalar_M_u_Inverse(*samples[s], *vec, scalar);
        }
    }

    void Precompute_W_u_Plus_scalar_M_u_Data(RealT &scalar) override
    {
        RealT scalar_shift = scalar * u_hyperparam_interface_->Get_alpha_u();
        scalars_.push_back(scalar_shift);

        HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> A = W_u_acute_->Clone();
        A->Set(*W_u_acute_);
        A->Scaled_Plus(scalar_shift, *M_);
        A->Set_Symmetric();
        W_u_acute_plus_scalar_M_u_.push_back(A);

        std::string A_solver_message = "W_u_acute_plus_scalar_M_u_Inverse";
        HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> A_solver = A->Get_Sparse_Matrix_Solver(use_direct_solvers_, verbosity_, out_stream_, A_solver_message);
        std::string A_sqrt_solver_message = "W_u_acute_plus_scalar_M_u_Sqrt";
        HDSA::Ptr<HDSA::Sparse_Matrix_Sqrt<RealT>> A_sqrt = HDSA::makePtr<HDSA::Sparse_Matrix_Sqrt<RealT>>(A, A_sqrt_solver_message);

        if (use_incomplete_prec_)
        {
            if (verbosity_ > 2)
            {
                timer_->Start_Timer();
            }
            HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = A->Get_Incomplete_Chol_Factor();
            A_solver->Set_Incomplete_Factor(L);
            A_sqrt->Set_Incomplete_Factor(L);
            if (verbosity_ > 2)
            {
                RealT elaped_time = timer_->End_Timer();
                out_stream_ << "Shifted W_u_acute incomplete factorization took " << elaped_time << " seconds." << std::endl;
            }
        }

        W_u_acute_plus_scalar_M_u_solver_.push_back(A_solver);
        W_u_acute_plus_scalar_M_u_sqrt_.push_back(A_sqrt);
    }

    void Disable_Sampling_Preconditioner(void)
    {
        for (long unsigned int k = 0; k < W_u_acute_plus_scalar_M_u_sqrt_.size(); k++)
        {
            W_u_acute_plus_scalar_M_u_sqrt_[k]->Disable_Incomplete_Factorization();
        }
    }

    MD_Lumped_Mass_u_Prior_Interface(const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> &S, const HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> &M, const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface,
                                     const HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> &u_hyperparam_interface, const bool use_direct_solvers = false, const int verbosity = 0,
                                     const bool use_incomplete_prec = true, std::ostream &out_stream = std::cout)
        : HDSA::MD_Scaled_u_Prior_Interface<RealT>(u_hyperparam_interface->Get_alpha_u()), S_(S), M_(M), data_interface_(data_interface), u_hyperparam_interface_(u_hyperparam_interface),
          use_direct_solvers_(use_direct_solvers), verbosity_(verbosity), use_incomplete_prec_(use_incomplete_prec), out_stream_(out_stream)
    {
        timer_ = HDSA::makePtr<HDSA::Timer<RealT>>(out_stream_);
        if (HDSA::Ptr<const HDSA::Transient_Vector<RealT>> u_opt_trans = HDSA::dynamicPtrCast<const HDSA::Transient_Vector<RealT>>(data_interface_->Get_u_opt()))
        {
            M_lumped_ = data_interface_->Extract_State_Component(*(*u_opt_trans)[0], u_hyperparam_interface_->Get_Component_ID())->Clone();
        }
        else
        {
            M_lumped_ = data_interface_->Extract_State_Component(*data_interface_->Get_u_opt(), u_hyperparam_interface_->Get_Component_ID())->Clone();
        }
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = M_lumped_->Clone();
        u_tmp->Set_Scalar(1.0);
        M_->Apply(*M_lumped_, *u_tmp);

        E_u_ = M_->Clone();
        determine_u_hyperparams_ = HDSA::makePtr<HDSA::MD_Determine_u_Hyperparameters<RealT>>(data_interface_, u_hyperparam_interface_);

        if (u_hyperparam_interface_->Get_beta_u() == 0.0)
        {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error, "Error in HDSA::MD_Numeric_Lumped_Mass_u_Prior_Interface: The value of beta_u must be specificed" << std::endl);
        }
        Set_beta_u(u_hyperparam_interface_->Get_beta_u());

        if (!u_hyperparam_interface_->Is_Transient())
        {
            if (u_hyperparam_interface_->Get_alpha_u() == 0.0)
            {
                determine_u_hyperparams_->Determine_alpha_u(this);
            }
            this->Set_alpha_u(u_hyperparam_interface_->Get_alpha_u());
        }

        if (u_hyperparam_interface_->Get_alpha_d() == 0.0)
        {
            determine_u_hyperparams_->Determine_alpha_d(this);
        }
    }

    virtual ~MD_Lumped_Mass_u_Prior_Interface()
    {
    }

    void Set_beta_u(RealT beta_u_new)
    {
        beta_u_ = beta_u_new;
        E_u_->Set(*M_);
        E_u_->Scaled_Plus(beta_u_new, *S_);

        E_u_->Set_Symmetric();
        std::string A_solver_message = "E_u_Inverse";
        E_u_solver_ = E_u_->Get_Sparse_Matrix_Solver(use_direct_solvers_, verbosity_, out_stream_, A_solver_message);
        if (use_incomplete_prec_)
        {
            if (verbosity_ > 3)
            {
                timer_->Start_Timer();
            }
            HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = E_u_->Get_Incomplete_Chol_Factor();
            E_u_solver_->Set_Incomplete_Factor(L);
            if (verbosity_ > 3)
            {
                RealT elaped_time = timer_->End_Timer();
                out_stream_ << "E_u incomplete factorization took " << elaped_time << " seconds." << std::endl;
            }
        }
        Assemble_W_u_Acute();
    }

    void Apply_E_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
    {
        if (verbosity_ > 3)
        {
            timer_->Start_Timer();
        }
        std::string output_message = E_u_solver_->Apply_A_Inverse(u_out, u_in);
        if (verbosity_ > 3)
        {
            RealT elapsed_time = timer_->End_Timer();
            out_stream_ << output_message << " in " << elapsed_time << " seconds." << std::endl;
        }
    }

    void Assemble_W_u_Acute(void)
    {
        int max_entries_per_row = 3 * E_u_->Get_Max_Nonzeros_Per_Row(); // This may fail in spatial dimensions > 2. Need to
                                                                        // test this and possibly modify.

        HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> D_sm = E_u_->Clone(1);
        D_sm->Set_Diagonal(*M_lumped_, true);
        HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> tmp = E_u_->Clone(max_entries_per_row);
        W_u_acute_ = E_u_->Clone(max_entries_per_row);

        E_u_->Matrix_Matrix_Multiply(*tmp, *D_sm, true, false);
        tmp->Matrix_Matrix_Multiply(*W_u_acute_, *E_u_, false, false);
    }
};
} // namespace HDSA

#include "HDSA_MD_Determine_u_Hyperparameters_Def.hpp"

#endif
