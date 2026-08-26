/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_RANDOMIZED_GSVD_HPP
#define HDSA_RANDOMIZED_GSVD_HPP

#include "HDSA_Linear_Algebra.hpp"

// This class executes the randomized GSVD solver

namespace HDSA
{

  template <class RealT>
  class Randomized_GSVD
  {

  public:
    Randomized_GSVD() {}

    virtual ~Randomized_GSVD() {}

    virtual void Apply_Operator(HDSA::Vector<RealT> &vec_out, const HDSA::Vector<RealT> &vec_in) const = 0;

    virtual void Apply_Operator_Transpose(HDSA::Vector<RealT> &vec_out, const HDSA::Vector<RealT> &vec_in) const = 0;

    virtual void Apply_Input_Weighting_Operator_Inverse(HDSA::Vector<RealT> &vec_out, const HDSA::Vector<RealT> &vec_in) const = 0;

    virtual void Apply_Output_Weighting_Operator(HDSA::Vector<RealT> &vec_out, const HDSA::Vector<RealT> &vec_in) const = 0;

    virtual void Generate_Random_Samples(HDSA::MultiVector<RealT> &samples) const = 0;

    void Compute_GSVD(HDSA::MultiVector<RealT> &sing_vecs_input, HDSA::MultiVector<RealT> &sing_vecs_output, HDSA::Dense_Matrix<RealT> &sing_vals,
                      int num_sing_vals, int oversampling, int num_subspace_iters)
    {
      int kpp = num_sing_vals + oversampling;

      HDSA::Ptr<HDSA::MultiVector<RealT>> samples = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
      Generate_Random_Samples(*samples);
      HDSA::Ptr<HDSA::MultiVector<RealT>> Y = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_output[0]);
      for (int k = 0; k < kpp; k++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> vec_in_random = (*samples)[k];
        Apply_Operator(*(*Y)[k], *vec_in_random);
      }

      HDSA::Ptr<HDSA::MultiVector<RealT>> Q = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_output[0]);
      HDSA::Ptr<HDSA::MultiVector<RealT>> WQ = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_output[0]);
      std::string type = "output_weighting";
      CholQR(*Q, *WQ, *Y, type);

      for (int j = 0; j < num_subspace_iters; j++)
      {
        HDSA::Ptr<HDSA::MultiVector<RealT>> Y_subspace_iter_in = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
        for (int k = 0; k < kpp; k++)
        {
          Apply_Operator_Transpose(*(*Y_subspace_iter_in)[k], *(*WQ)[k]);
        }
        type = "input_weighting_inverse";
        HDSA::Ptr<HDSA::MultiVector<RealT>> Q_iter_in = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
        HDSA::Ptr<HDSA::MultiVector<RealT>> WQ_iter_in = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
        CholQR(*Q_iter_in, *WQ_iter_in, *Y_subspace_iter_in, type);

        Y->Zeros();
        for (int k = 0; k < kpp; k++)
        {
          Apply_Operator(*(*Y)[k], *(*WQ_iter_in)[k]);
        }
        type = "output_weighting";
        Q->Zeros();
        WQ->Zeros();
        CholQR(*Q, *WQ, *Y, type);
      }

      HDSA::Ptr<HDSA::MultiVector<RealT>> B = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
      HDSA::Ptr<HDSA::MultiVector<RealT>> Tinv_B = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
      for (int k = 0; k < kpp; k++)
      {
        Apply_Operator_Transpose(*(*B)[k], *(*WQ)[k]);
        Apply_Input_Weighting_Operator_Inverse(*(*Tinv_B)[k], *(*B)[k]);
      }
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> C = Tinv_B->MatMat(*B);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_B = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, kpp);
      HDSA::Linear_Algebra::Cholesky_Factorization<RealT>(*C, *R_B);

      HDSA::Ptr<HDSA::MultiVector<RealT>> Q_B = HDSA::makePtr<HDSA::MultiVector<RealT>>(kpp, *sing_vecs_input[0]);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> I = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, kpp);
      for (int k = 0; k < kpp; k++)
      {
        I->Set_Entry(k, k, 1.0);
      }
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_B_inv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, kpp);
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(*R_B_inv, *I, *R_B);
      for (int k = 0; k < kpp; k++)
      {
        for (int i = 0; i < kpp; i++)
        {
          (*Q_B)[k]->Scaled_Plus((*R_B_inv)(i, k), *(*Tinv_B)[i]);
        }
      }

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> UT = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, kpp);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> V = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, kpp);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(kpp, 1);
      HDSA::Linear_Algebra::SVD<RealT>(*R_B, *V, *UT, *S);

      sing_vecs_input.Zeros();
      sing_vecs_output.Zeros();
      for (int k = 0; k < num_sing_vals; k++)
      {

        for (int i = 0; i < kpp; i++)
        {
          sing_vecs_input[k]->Scaled_Plus((*V)(i, k), *(*Q_B)[i]);
          sing_vecs_output[k]->Scaled_Plus((*UT)(k, i), *(*Q)[i]);
          sing_vals.Set_Entry(k, 0, (*S)(k, 0));
        }

        if ((*V)(0, k) < 0.0)
        {
          sing_vecs_input[k]->Scale(-1.0);
          sing_vecs_output[k]->Scale(-1.0);
        }
      }
    }

    void CholQR(HDSA::MultiVector<RealT> &Q, HDSA::MultiVector<RealT> &WQ, const HDSA::MultiVector<RealT> &Z, std::string &type)
    {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> ZtZ = Z.MatMat(Z);
      int n = ZtZ->Number_of_Columns();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_Z = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      HDSA::Linear_Algebra::Cholesky_Factorization<RealT>(*ZtZ, *R_Z);

      HDSA::Ptr<HDSA::MultiVector<RealT>> Q_Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(n, *Z[0]);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> I = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      for (int k = 0; k < n; k++)
      {
        I->Set_Entry(k, k, 1.0);
      }
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_Z_inv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(*R_Z_inv, *I, *R_Z);
      for (int k = 0; k < n; k++)
      {
        for (int i = 0; i < n; i++)
        {
          (*Q_Z)[k]->Scaled_Plus((*R_Z_inv)(i, k), *Z[i]);
        }
      }
      // Results in the factorization Z = Q_Z*R_Z

      // Applying weighting matrix to Q_Z
      HDSA::Ptr<HDSA::MultiVector<RealT>> W_Q_Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(n, *Z[0]);
      for (int k = 0; k < n; k++)
      {
        if (type == "input_weighting_inverse")
        {
          Apply_Input_Weighting_Operator_Inverse(*(*W_Q_Z)[k], *(*Q_Z)[k]);
        }
        else if (type == "output_weighting")
        {
          Apply_Output_Weighting_Operator(*(*W_Q_Z)[k], *(*Q_Z)[k]);
        }
        else
        {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA::Randomized_GSVD::CholQR incorrect type specification for CholQR" << std::endl);
        }
      }

      // Compute C = Z^T*W*Z = Z^T*X
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> C = Q_Z->MatMat(*W_Q_Z);

      // Compute R_C=chol(C)
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_C = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      HDSA::Linear_Algebra::Cholesky_Factorization<RealT>(*C, *R_C);

      // Compute Q=Q_Z*R_C^{-1}
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R_C_inv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(*R_C_inv, *I, *R_C);

      for (int k = 0; k < n; k++)
      {
        for (int i = 0; i < n; i++)
        {
          Q[k]->Scaled_Plus((*R_C_inv)(i, k), *(*Q_Z)[i]);
          WQ[k]->Scaled_Plus((*R_C_inv)(i, k), *(*W_Q_Z)[i]);
        }
      }
    }
  };

}

#endif
