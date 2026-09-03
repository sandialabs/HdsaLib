/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_PSEUDO_TIME_CONTINUATION_HPP
#define HDSA_PC_PSEUDO_TIME_CONTINUATION_HPP

#include "HDSA_Vector.hpp"
#include "HDSA_PC_Sensitivity_Operator_Interface.hpp"
#include "HDSA_PC_Quasi_Newton_Preconditioner.hpp"

namespace HDSA
{

	template <class RealT>
	class PC_Pseudo_Time_Continuation
	{

	private:
		HDSA::Ptr<HDSA::Vector<RealT>> z_bar_;
		HDSA::Ptr<HDSA::PC_Sensitivity_Operator_Interface<RealT>> sen_op_interface_;
		HDSA::Ptr<HDSA::PC_Quasi_Newton_Preconditioner<RealT>> qn_prec_;
		RealT grad_tol_;
		bool use_qn_prec_;
		bool print_cg_output_;
		bool print_cg_iter_;
		RealT cg_tol_;
		int max_cg_iter_;
		int max_extra_newton_;

	protected:
		virtual int Apply_Inverse_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index, RealT &cg_tol) const
		{

			z_out.Zeros();
			RealT z_in_Norm = z_in.Norm();
			if (z_in_Norm == 0.0) { return 0; }
			HDSA::Ptr<HDSA::Vector<RealT>> r = z_in.Clone();
			r->Set(z_in);
			r->Scale(1.0 / z_in_Norm);
			HDSA::Ptr<HDSA::Vector<RealT>> v = z_in.Clone();
			qn_prec_->Apply_Inverse_Hessian_Approximation(*v, *r);
			HDSA::Ptr<HDSA::Vector<RealT>> p = z_in.Clone();
			p->Set(*v);
			RealT scalar = r->Dot(*p);

			RealT r_Norm = r->Norm();
			int iter = 0;
			HDSA::Ptr<HDSA::Vector<RealT>> w = z_in.Clone();

      //avoid check on (std::sqrt(scalar) > cg_tol) because if false at the beginning, the function would return z_out=0
			while ((r_Norm > cg_tol) && (iter < max_cg_iter_))
			{
				iter += 1;
				sen_op_interface_->Apply_Hessian(*w, *p, z, theta_traj, time_index);
				qn_prec_->Add_Block_Quasi_Newton_Step(p, w);

				RealT scalar_tmp = w->Dot(*p);
				RealT alpha = scalar / scalar_tmp;
				z_out.Scaled_Plus(alpha, *p);
				r->Scaled_Plus(-alpha, *w);
				qn_prec_->Apply_Inverse_Hessian_Approximation(*v, *r);
				RealT scalar_old = scalar;
				scalar = v->Dot(*r);
				p->Scale(scalar / scalar_old);
				p->Plus(*v);
				r_Norm = r->Norm();

				if (print_cg_iter_)
				{
					std::cout << "Iteration = " << iter << " with relative residual = " << r_Norm << std::endl;
				}
			}

			z_out.Scale(z_in_Norm);

			if (print_cg_output_)
			{
				std::cout << "Total iterations = " << iter << std::endl;
				std::cout << "Relative residual = " << r_Norm << std::endl;
			}

			return iter;
		}

	public:
		PC_Pseudo_Time_Continuation(const HDSA::Ptr<HDSA::Vector<RealT>> &z_bar, const HDSA::Ptr<HDSA::PC_Sensitivity_Operator_Interface<RealT>> &sen_op_interface, const HDSA::Ptr<HDSA::PC_Quasi_Newton_Preconditioner<RealT>> &qn_prec,
									const RealT grad_tol, const bool use_qn_prec = true, const bool print_cg_output = true, const bool print_cg_iter = false, const RealT cg_tol = 1.e-5, const int max_cg_iter = 100) : z_bar_(z_bar), sen_op_interface_(sen_op_interface), qn_prec_(qn_prec), grad_tol_(grad_tol), use_qn_prec_(use_qn_prec), print_cg_output_(print_cg_output), print_cg_iter_(print_cg_iter), cg_tol_(cg_tol), max_cg_iter_(max_cg_iter)
		{
			max_extra_newton_ = 50;
		}

		virtual ~PC_Pseudo_Time_Continuation()
		{
		}

		void Pseudo_Time_Continuation_Forward_Euler(HDSA::Vector<RealT> &z_star, HDSA::Vector<RealT> &grad_star, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj)
		{
			HDSA::Ptr<HDSA::Vector<RealT>> z_current = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> z_new = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> grad_current = z_star.Clone();

			int num_Hvecs = 0;
			int num_Bvecs = 0;
			int num_grads = 0;

			if (use_qn_prec_)
			{
				num_Hvecs += qn_prec_->Get_Number_HessVecs();
			}

			int N = theta_traj.Get_Number_of_Timesteps();
			RealT dt = 1.0 / static_cast<RealT>(N);
			RealT time_index = 0.0;

			z_current->Set(*z_bar_);
			sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
			num_grads += 1;

			HDSA::Ptr<HDSA::Vector<RealT>> s, y;
			if (use_qn_prec_)
			{
				qn_prec_->Set_N(2 * N + 1);
				s = z_new->Clone();
				y = z_new->Clone();
			}

			for (int k = 0; k < N; k++)
			{
				RealT sol_grad_norm = grad_current->Norm();
				if (sol_grad_norm > grad_tol_)
				{
					std::cout << "Pseudo_Time_Continuation_Forward_Euler: Failed to converge! (" << sol_grad_norm << " > " << grad_tol_ << ")" << std::endl;
          exit(1);
					break;
				}
				std::cout << "-----------------------------------------------------" << std::endl;
				std::cout << "Pseudo_Time_Continuation_Forward_Euler: The gradient Norm after step: " << k << " is " << sol_grad_norm << std::endl;
				std::cout << "Pseudo_Time_Continuation_Forward_Euler: Beginning B matvec at time step " << k + 1 << std::endl;
				time_index = static_cast<RealT>(k);
				sen_op_interface_->Apply_B(*z_tmp, *z_current, theta_traj, time_index);
				num_Bvecs += 1;

				std::cout << "Pseudo_Time_Continuation_Forward_Euler: Beginning inverse Hessian matvec at Euler step " << k + 1 << std::endl;
				int iters = Apply_Inverse_Hessian(*z_new, *z_tmp, *z_current, theta_traj, time_index, cg_tol_);
				num_Hvecs += iters;

				z_new->Scale(-dt);
				z_new->Plus(*z_current);

				if (use_qn_prec_)
				{
					s->Set(*z_new);
					s->Scaled_Plus(-1.0, *z_current);

					y->Set(*grad_current);
					y->Scaled_Plus(dt, *z_tmp);
					y->Scale(-1.0);
				}
				z_current->Set(*z_new);

				time_index = static_cast<RealT>(k+1);
				sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
				num_grads += 1;
				if (use_qn_prec_ & (k < N - 1))
				{
					qn_prec_->Add_Block_Quasi_Newton_Data();
					y->Plus(*grad_current);
					qn_prec_->Add_Parametric_Quasi_Newton_Data(*s, *y);
				}
				sol_grad_norm = grad_current->Norm();

				HDSA::Ptr<HDSA::Vector<RealT>> z_old = z_current->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> grad_trial = grad_current->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> H_step = z_current->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> A_step = z_current->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> lin_res = z_current->Clone();

        const RealT armijo_c = 1.e-4;
        const RealT backtrack_factor = 0.5;
        const RealT eta_max = 0.5;
        const int max_backtracks = 20;
        const int max_cg_retries = 4;

        const RealT eta_forcing_max = 0.1;
        RealT variable_cg_tol = eta_forcing_max;
        int num_extra_newton = 0;
        RealT initial_grad_norm = sol_grad_norm;

        //Newton loop to find the zero of the solution gradient g(z), using backtracking and armijo condition based on minimizing ||g(z)||^2.
        while ((sol_grad_norm > grad_tol_) && (num_extra_newton < max_extra_newton_))
        {
          std::cout << "Pseudo_Time_Continuation_Forward_Euler: Newton iteration at step " << k + 1 << std::endl;
          std::cout << "Pseudo_Time_Continuation_Forward_Euler: The current gradient Norm = " << sol_grad_norm << std::endl;

          const RealT old_grad_norm = sol_grad_norm;
          const RealT old_grad_norm_sq = old_grad_norm * old_grad_norm;

          // Clone() does not copy values.
          z_old->Set(*z_current);

          bool direction_ok = false;
          RealT directional_derivative = 0.0;
          RealT eta_actual = 0.0;
          RealT cg_tol_try = variable_cg_tol;

          for (int cg_retry = 0; cg_retry < max_cg_retries; ++cg_retry)
          {
            // Approximately solve A p = g.
            iters = Apply_Inverse_Hessian(*z_new, *grad_current, *z_current, theta_traj, time_index, cg_tol_try);
            num_Hvecs += iters;

            // Compute H p, where H is the derivative of the gradient.
            sen_op_interface_->Apply_Hessian(*H_step, *z_new, *z_current, theta_traj, time_index);
            num_Hvecs += 1;

            // Reconstruct the operator used by CG: A p = H p + quasi-Newton correction.
            A_step->Set(*H_step);
            if (use_qn_prec_)
              qn_prec_->Add_Block_Quasi_Newton_Step(z_new, A_step);

            // Linear residual r = g - A p.
            lin_res->Set(*grad_current);
            lin_res->Scaled_Plus(-1.0, *A_step);
            eta_actual = lin_res->Norm() / old_grad_norm;

            // For phi(z) = 0.5 ||g(z)||^2 and d = -p:
            // phi'(0) = -g^T H p.
            directional_derivative = -grad_current->Dot(*H_step);

            std::cout << "  CG relative residual = " << eta_actual
                      << ", phi'(0) = " << directional_derivative << std::endl;

            if ((eta_actual <= eta_max) && (directional_derivative < 0.0))
            {
              direction_ok = true;
              break;
            }

            cg_tol_try *= 0.1;
            std::cout << "  Recomputing Newton direction with CG tolerance " << cg_tol_try << std::endl;
          }

          if (!direction_ok)
          {
            std::cout << "Failed to obtain a sufficiently accurate descent Newton direction." << std::endl;
            z_current->Set(*z_old);
            break;
          }

          RealT alpha = 1.0;
          bool accepted = false;
          RealT min_bt_grad_norm = initial_grad_norm;
          RealT best_alpha = 1.0;

          for (int bt = 0; bt < max_backtracks; ++bt)
          {
            // z_trial = z_old - alpha p.
            z_current->Set(*z_old);
            z_current->Scaled_Plus(-alpha, *z_new);

            sen_op_interface_->Gradient(*grad_trial, *z_current, theta_traj, time_index);
            num_grads += 1;

            const RealT trial_grad_norm = grad_trial->Norm();
            if(trial_grad_norm < min_bt_grad_norm) {
              min_bt_grad_norm = trial_grad_norm;
              best_alpha = alpha;
            }
            const RealT trial_grad_norm_sq = trial_grad_norm * trial_grad_norm;

            // Armijo condition for phi(z) = 0.5 ||g(z)||^2.
            const RealT armijo_rhs = old_grad_norm_sq + 2.0 * armijo_c * alpha * directional_derivative;

            std::cout << "  Backtracking alpha = " << alpha
                      << ", ||g_trial|| = " << trial_grad_norm << std::endl;

            if (trial_grad_norm_sq <= armijo_rhs)
            {
              grad_current->Set(*grad_trial);
              sol_grad_norm = trial_grad_norm;
              accepted = true;
              break;
            }

            alpha *= backtrack_factor;
          }

          if (accepted) {
            std::cout << "Accepted Newton step with alpha = " << alpha
                      << ", new gradient Norm = " << sol_grad_norm << std::endl;

            const RealT relative_grad = sol_grad_norm / initial_grad_norm;
            variable_cg_tol = std::max(cg_tol_, std::min(eta_forcing_max, std::sqrt(relative_grad)));
          } else {
            if(alpha != best_alpha){
              z_current->Set(*z_old);
              z_current->Scaled_Plus(-best_alpha, *z_new);
              sen_op_interface_->Gradient(*grad_trial, *z_current, theta_traj, time_index);
              grad_current->Set(*grad_trial);
            }
            sol_grad_norm = min_bt_grad_norm;
            //sol_grad_norm = old_grad_norm;
            std::cout << "Newton backtracking failed. Using recovery step" << std::endl;
            //break;
          }          

          num_extra_newton += 1;
        }
			}

			z_star.Set(*z_current);
			grad_star.Set(*grad_current);

			RealT sol_grad_Norm = grad_current->Norm();
			std::cout << " " << std::endl;
			std::cout << "-----------------------------------------------------" << std::endl;
			std::cout << "Solution gradient Norm = " << sol_grad_Norm << std::endl;
			std::cout << " " << std::endl;

			std::string name = "Forward_Euler_Cost_Report.txt";
			std::ofstream fout;
			fout.open(name);
			fout << "Number of B-vector products: " << num_Bvecs << std::endl;
			fout << "Number of H-vector products: " << num_Hvecs << std::endl;
			fout << "Number of gradient evaluations: " << num_grads << std::endl;
			fout << "Number of vectors stored for preconditioner: " << qn_prec_->Get_Number_Vecs_Stored() << std::endl;
			fout << "Solution gradient Norm = " << sol_grad_Norm << std::endl;
			fout.close();
		}

		void Pseudo_Time_Continuation_Modified_Euler(HDSA::Vector<RealT> &z_star, HDSA::Vector<RealT> &grad_star, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj)
		{
			HDSA::Ptr<HDSA::Vector<RealT>> z_current = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> z_new = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> z_store = z_star.Clone();
			HDSA::Ptr<HDSA::Vector<RealT>> grad_current = z_star.Clone();

			int num_Hvecs = 0;
			int num_Bvecs = 0;
			int num_grads = 0;

			if (use_qn_prec_)
			{
				num_Hvecs += qn_prec_->Get_Number_HessVecs();
			}

			int N = theta_traj.Get_Number_of_Timesteps();
			RealT dt = 1.0 / static_cast<RealT>(N);
			RealT time_index = 0.0;

			z_current->Set(*z_bar_);
			sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
			num_grads += 1;

			HDSA::Ptr<HDSA::Vector<RealT>> s, y;
			if (use_qn_prec_)
			{
				qn_prec_->Set_N(3 * N + 1);
				s = z_new->Clone();
				y = z_new->Clone();
			}

			for (int k = 0; k < N; k++)
			{
				RealT sol_grad_Norm = grad_current->Norm();
				if (sol_grad_Norm > grad_tol_)
				{
					std::cout << "Pseudo_Time_Continuation_Modified_Euler: Failed to converge! (" << sol_grad_Norm << " > " << grad_tol_ << ")" << std::endl;
          exit(1);
					break;
				}
				std::cout << "-----------------------------------------------------" << std::endl;
				std::cout << "Pseudo_Time_Continuation_Modified_Euler: The gradient Norm after step: " << k << " is " << grad_current->Norm() << std::endl;

				z_store->Set(*z_current);

				std::cout << "Pseudo_Time_Continuation_Modified_Euler: Beginning B matvec at Euler step " << k + 1 << std::endl;
				time_index = static_cast<RealT>(k);
				sen_op_interface_->Apply_B(*z_tmp, *z_current, theta_traj, time_index);
				num_Bvecs += 1;

				std::cout << "Pseudo_Time_Continuation_Modified_Euler: Beginning inverse Hessian matvec at Euler step " << k + 1 << std::endl;
				int iters = Apply_Inverse_Hessian(*z_new, *z_tmp, *z_current, theta_traj, time_index, cg_tol_);
				num_Hvecs += iters;

				z_new->Scale(-0.5 * dt);
				z_new->Plus(*z_current);

				if (use_qn_prec_)
				{
					s->Set(*z_new);
					s->Scaled_Plus(-1.0, *z_current);

					y->Set(*grad_current);
					y->Scaled_Plus(0.5 * dt, *z_tmp);
					y->Scale(-1.0);
				}

				z_current->Set(*z_new);
				z_new->Zeros();

				time_index = static_cast<RealT>(k) + 0.5;
				sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
				num_grads += 1;
				if (use_qn_prec_)
				{
					qn_prec_->Add_Block_Quasi_Newton_Data();
					y->Plus(*grad_current);
					qn_prec_->Add_Parametric_Quasi_Newton_Data(*s, *y);
				}

				std::cout << "Pseudo_Time_Continuation_Modified_Euler: Beginning B matvec at Euler step " << k + 1 << " 1/2" << std::endl;
				sen_op_interface_->Apply_B(*z_tmp, *z_current, theta_traj, time_index);
				num_Bvecs += 1;

				std::cout << "Pseudo_Time_Continuation_Modified_Euler: Beginning inverse Hessian matvec at Euler step " << k + 1 << " 1/2" << std::endl;
				iters = Apply_Inverse_Hessian(*z_new, *z_tmp, *z_current, theta_traj, time_index, cg_tol_);
				num_Hvecs += iters;

				z_new->Scale(-dt);
				z_new->Plus(*z_store);

				if (use_qn_prec_)
				{
					s->Set(*z_new);
					s->Scaled_Plus(-1.0, *z_current);

					y->Set(*grad_current);
					y->Scaled_Plus(0.5 * dt, *z_tmp);
					y->Scale(-1.0);
				}

				z_current->Set(*z_new);
				time_index = static_cast<RealT>(k+1);
				sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
				num_grads += 1;

				if (use_qn_prec_)
				{
					qn_prec_->Add_Block_Quasi_Newton_Data();
					y->Plus(*grad_current);
					qn_prec_->Add_Parametric_Quasi_Newton_Data(*s, *y);
				}
				sol_grad_Norm = grad_current->Norm();

				if (sol_grad_Norm > grad_tol_)
				{
					std::cout << "Pseudo_Time_Continuation_Modified_Euler: Beginning inverse Hessian matvec at Newton step " << k + 1 << std::endl;
					std::cout << "Pseudo_Time_Continuation_Modified_Euler: The current gradient Norm = " << sol_grad_Norm << std::endl;
					iters = Apply_Inverse_Hessian(*z_new, *grad_current, *z_current, theta_traj, time_index, cg_tol_);
					num_Hvecs += iters;

					z_current->Scaled_Plus(-1.0, *z_new);
					sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
					num_grads += 1;

					if (use_qn_prec_)
					{
						qn_prec_->Add_Block_Quasi_Newton_Data();
					}

					sol_grad_Norm = grad_current->Norm();
				}

				RealT variable_cg_tol = cg_tol_;
				int num_extra_newton = 0;
				while (sol_grad_Norm > grad_tol_ && num_extra_newton < max_extra_newton_)
				{
					std::cout << "Pseudo_Time_Continuation_Modified_Euler: Taking an extra Newton iteration at step " << k + 1 << std::endl;
					std::cout << "Pseudo_Time_Continuation_Modified_Euler: The current gradient Norm = " << sol_grad_Norm << std::endl;
					iters = Apply_Inverse_Hessian(*z_new, *grad_current, *z_current, theta_traj, time_index, variable_cg_tol);
					num_Hvecs += iters;
					z_current->Scaled_Plus(-1.0, *z_new);
					sen_op_interface_->Gradient(*grad_current, *z_current, theta_traj, time_index);
					num_grads += 1;
					sol_grad_Norm = grad_current->Norm();
					variable_cg_tol = (1.e-1) * variable_cg_tol;
					num_extra_newton += 1;
				}
			}
			z_star.Set(*z_current);
			grad_star.Set(*grad_current);

			RealT sol_grad_Norm = grad_current->Norm();
			std::cout << " " << std::endl;
			std::cout << "-----------------------------------------------------" << std::endl;
			std::cout << "Solution gradient Norm = " << sol_grad_Norm << std::endl;
			std::cout << " " << std::endl;

			std::string name = "Modified_Euler_Cost_Report.txt";
			std::ofstream fout;
			fout.open(name);
			fout << "Number of B-vector products: " << num_Bvecs << std::endl;
			fout << "Number of H-vector products: " << num_Hvecs << std::endl;
			fout << "Number of gradient evaluations: " << num_grads << std::endl;
			fout << "Number of vectors stored for preconditioner: " << qn_prec_->Get_Number_Vecs_Stored() << std::endl;
			fout << "Solution gradient Norm = " << sol_grad_Norm << std::endl;
			fout.close();
		}
	};

}

#endif
