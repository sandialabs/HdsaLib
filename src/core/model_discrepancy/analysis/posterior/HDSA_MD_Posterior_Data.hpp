/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_POSTERIOR_DATA_HPP
#define HDSA_MD_POSTERIOR_DATA_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"

namespace HDSA
{

	template <class RealT>
	class MD_Posterior_Data
	{

	public:
		RealT alpha_d;
		int N;
		HDSA::Ptr<HDSA::MultiVector<RealT>> M_z_Z;
		HDSA::Ptr<HDSA::MultiVector<RealT>> W_z_inv_M_z_Z;
		HDSA::Ptr<HDSA::MultiVector<RealT>> M_z_W_z_inv_M_z_Z;
		HDSA::Ptr<HDSA::Vector<RealT>> M_z_z_opt;
		HDSA::Ptr<HDSA::Vector<RealT>> W_z_inv_M_z_z_opt;
		HDSA::Ptr<HDSA::Vector<RealT>> M_z_W_z_inv_M_z_z_opt;
		HDSA::Ptr<HDSA::MultiVector<RealT>> Zc;
		HDSA::Ptr<HDSA::MultiVector<RealT>> M_z_Zc;
		HDSA::Ptr<HDSA::MultiVector<RealT>> W_z_inv_M_z_Zc;
		HDSA::Ptr<HDSA::MultiVector<RealT>> M_z_W_z_inv_M_z_Zc;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Zc_M_z_W_z_inv_M_z_Zc;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> G;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> g_vecs;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Mu;
		std::vector<RealT> sum_g_vecs;
		HDSA::Ptr<HDSA::MultiVector<RealT>> u_ell;
		std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> u_i_ell;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> a_ell;
		HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b_i_ell;
		int num_samples;
		std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> u_i_hat;
		HDSA::Ptr<HDSA::MultiVector<RealT>> u_breve;
		HDSA::Ptr<HDSA::MultiVector<RealT>> z_breve;
		HDSA::Ptr<HDSA::MultiVector<RealT>> M_z_z_breve;

		MD_Posterior_Data(void)
		{
		}

		~MD_Posterior_Data(void)
		{
		}

		void Compute_Posterior_Data(HDSA::MD_Data_Interface<RealT> &data_interface, HDSA::MD_u_Prior_Interface<RealT> &u_prior_interface, const HDSA::MD_z_Prior_Interface<RealT> &z_prior_interface, const RealT alpha_d_in, int num_samples_in)
		{
			alpha_d = alpha_d_in;
			num_samples = num_samples_in;
			N = data_interface.Get_Z()->Number_of_Vectors();

			M_z_Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(N, *data_interface.Get_z_opt());
			W_z_inv_M_z_Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(N, *data_interface.Get_z_opt());
			M_z_W_z_inv_M_z_Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(N, *data_interface.Get_z_opt());
			for (int k = 0; k < N; k++)
			{
				HDSA::Ptr<HDSA::Vector<RealT>> zk = (*data_interface.Get_Z())[k];
				HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = (*M_z_Z)[k];
				z_prior_interface.Apply_M_z(*z_tmp1, *zk);
				HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = (*W_z_inv_M_z_Z)[k];
				z_prior_interface.Apply_W_z_Inverse(*z_tmp2, *z_tmp1);
				HDSA::Ptr<HDSA::Vector<RealT>> z_tmp3 = (*M_z_W_z_inv_M_z_Z)[k];
				z_prior_interface.Apply_M_z(*z_tmp3, *z_tmp2);
			}
			M_z_z_opt = data_interface.Get_z_opt()->Clone();
			W_z_inv_M_z_z_opt = data_interface.Get_z_opt()->Clone();
			M_z_W_z_inv_M_z_z_opt = data_interface.Get_z_opt()->Clone();
			z_prior_interface.Apply_M_z(*M_z_z_opt, *data_interface.Get_z_opt());
			z_prior_interface.Apply_W_z_Inverse(*W_z_inv_M_z_z_opt, *M_z_z_opt);
			z_prior_interface.Apply_M_z(*M_z_W_z_inv_M_z_z_opt, *W_z_inv_M_z_z_opt);

			Zc = HDSA::makePtr<HDSA::MultiVector<RealT>>(N - 1, *data_interface.Get_z_opt());
			M_z_Zc = HDSA::makePtr<HDSA::MultiVector<RealT>>(N - 1, *data_interface.Get_z_opt());
			W_z_inv_M_z_Zc = HDSA::makePtr<HDSA::MultiVector<RealT>>(N - 1, *data_interface.Get_z_opt());
			M_z_W_z_inv_M_z_Zc = HDSA::makePtr<HDSA::MultiVector<RealT>>(N - 1, *data_interface.Get_z_opt());
			for (int k = 0; k < N - 1; k++)
			{
				(*Zc)[k]->Set(*(*data_interface.Get_Z())[k + 1]);
				(*Zc)[k]->Scaled_Plus(-1.0, *data_interface.Get_z_opt());

				(*M_z_Zc)[k]->Set(*(*M_z_Z)[k + 1]);
				(*M_z_Zc)[k]->Scaled_Plus(-1.0, *M_z_z_opt);

				(*W_z_inv_M_z_Zc)[k]->Set(*(*W_z_inv_M_z_Z)[k + 1]);
				(*W_z_inv_M_z_Zc)[k]->Scaled_Plus(-1.0, *W_z_inv_M_z_z_opt);

				(*M_z_W_z_inv_M_z_Zc)[k]->Set(*(*M_z_W_z_inv_M_z_Z)[k + 1]);
				(*M_z_W_z_inv_M_z_Zc)[k]->Scaled_Plus(-1.0, *M_z_W_z_inv_M_z_z_opt);
			}
			Zc_M_z_W_z_inv_M_z_Zc = Zc->MatMat(*M_z_W_z_inv_M_z_Zc);

			G = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, N);
			RealT z_opt_M_z_W_z_inv_M_z_z_opt = data_interface.Get_z_opt()->Dot(*M_z_W_z_inv_M_z_z_opt);
			for (int i = 0; i < N; i++)
			{
				HDSA::Ptr<HDSA::Vector<RealT>> zi = (*data_interface.Get_Z())[i];
				HDSA::Ptr<HDSA::Vector<RealT>> gzi = (*M_z_W_z_inv_M_z_Z)[i];
				RealT vali = 1.0 + z_opt_M_z_W_z_inv_M_z_z_opt - zi->Dot(*M_z_W_z_inv_M_z_z_opt);
				for (int j = 0; j < i + 1; j++)
				{
					HDSA::Ptr<HDSA::Vector<RealT>> zj = (*data_interface.Get_Z())[j];
					RealT val = vali;
					val -= zj->Dot(*M_z_W_z_inv_M_z_z_opt);
					val += zj->Dot(*gzi);
					G->Set_Entry(i, j, val);
				}
			}
			for (int i = 0; i < N; i++)
			{
				for (int j = i + 1; j < N; j++)
				{
					G->Set_Entry(i, j, (*G)(j, i));
				}
			}

			g_vecs = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, N);
			Mu = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, 1);
			HDSA::Linear_Algebra::Symmetric_Eig_Decomposition<RealT>(*G, *g_vecs, *Mu);

			sum_g_vecs = std::vector<RealT>(N);
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					sum_g_vecs[i] += (*g_vecs)(j, i);
				}
			}

			u_ell = HDSA::makePtr<HDSA::MultiVector<RealT>>(N, *(*data_interface.Get_D())[0]);
			for (int ell = 0; ell < N; ell++)
			{
				HDSA::Ptr<HDSA::Vector<RealT>> dl = (*data_interface.Get_D())[ell];
				HDSA::Ptr<HDSA::Vector<RealT>> ul = (*u_ell)[ell];
				HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = ul->Clone();
				u_prior_interface.Apply_M_u(*u_tmp, *dl);
				u_prior_interface.Apply_W_u_Inverse(*ul, *u_tmp);
			}

			u_i_ell.resize(N);
			for (int i = 0; i < N; i++)
			{
				u_i_ell[i] = HDSA::makePtr<HDSA::MultiVector<RealT>>(N, *(*data_interface.Get_D())[0]);
				RealT scalar = (*Mu)(i, 0) / alpha_d;
				u_prior_interface.Precompute_W_u_Plus_scalar_M_u_Data(scalar);
				for (int ell = 0; ell < N; ell++)
				{
					HDSA::Ptr<HDSA::Vector<RealT>> uil = (*u_i_ell[i])[ell];
					HDSA::Ptr<HDSA::Vector<RealT>> ul = (*u_ell)[ell];
					HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = ul->Clone();
					u_prior_interface.Apply_M_u(*u_tmp, *ul);
					u_prior_interface.Apply_W_u_Plus_scalar_M_u_Inverse(*uil, *u_tmp, scalar);
					uil->Scale(1.0 / alpha_d);
				}
			}

			a_ell = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, 1);
			b_i_ell = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(N, N);
			for (int ell = 0; ell < N; ell++)
			{
				HDSA::Ptr<HDSA::Vector<RealT>> zl = (*data_interface.Get_Z())[ell];
				RealT val_a = 1.0 - zl->Dot(*M_z_W_z_inv_M_z_z_opt) + z_opt_M_z_W_z_inv_M_z_z_opt;
				a_ell->Set_Entry(ell, 0, val_a);
				for (int i = 0; i < N; i++)
				{
					RealT val_b = 0.0;
					for (int k = 0; k < N; k++)
					{
						HDSA::Ptr<HDSA::Vector<RealT>> gzk = (*M_z_W_z_inv_M_z_Z)[k];
						val_b += (*g_vecs)(k, i) * (zl->Dot(*gzk) - gzk->Dot(*data_interface.Get_z_opt()) + (*a_ell)(ell, 0));
					}
					b_i_ell->Set_Entry(i, ell, val_b);
				}
			}

			if (num_samples > 0)
			{
				u_i_hat.resize(N);
				for (int i = 0; i < N; i++)
				{
					u_i_hat[i] = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_samples, *(*data_interface.Get_D())[0]);
					u_prior_interface.Sample_with_Covariance_W_u_Plus_scalar_M_u_Inverse(*u_i_hat[i], (*Mu)(i, 0) / alpha_d);
					u_i_hat[i]->Scale(1.0 / std::sqrt(alpha_d));
				}

				u_breve = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_samples, *(*data_interface.Get_D())[0]);
				u_prior_interface.Sample_with_Covariance_W_u_Inverse(*u_breve);
				z_breve = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_samples, *(*data_interface.Get_Z())[0]);
				z_prior_interface.Sample_with_Covariance_W_z_Inverse(*z_breve);
				M_z_z_breve = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_samples, *(*data_interface.Get_Z())[0]);
				for (int k = 0; k < num_samples; k++)
				{
					HDSA::Ptr<HDSA::Vector<RealT>> z_breve_k = (*z_breve)[k];
					HDSA::Ptr<HDSA::Vector<RealT>> M_z_z_breve_k = (*M_z_z_breve)[k];
					z_prior_interface.Apply_M_z(*M_z_z_breve_k, *z_breve_k);
				}
			}
		}
	};

}

#endif
