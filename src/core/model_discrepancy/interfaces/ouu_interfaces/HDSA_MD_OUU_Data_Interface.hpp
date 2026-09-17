/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OUU_DATA_INTERFACE_HPP
#define HDSA_MD_OUU_DATA_INTERFACE_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Ensemble_Vector.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_OUU_Data_Interface : public HDSA::MD_Data_Interface<RealT>
  {

  private:
    int ens_size_;

  public:
    MD_OUU_Data_Interface(int ens_size) : ens_size_(ens_size)
    {
    }

    virtual ~MD_OUU_Data_Interface()
    {
    }

    int Get_Ensemble_Size(void) const
    {
      return ens_size_;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_us(int s) const = 0;

    virtual HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z(void) const = 0;

    virtual HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data(void) const = 0;

    virtual HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Ds_Data(int s) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // base class pure virtual function implementations
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u(void) const
    {
      std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> vecs;
      vecs.resize(ens_size_);
      for (int s = 0; s < ens_size_; s++)
      {
        vecs[s] = Load_Optimal_us(s);
      }
      HDSA::Ptr<HDSA::Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Ensemble_Vector<RealT>>(vecs);
      return u_opt;
    }

    HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data(void) const
    {
      std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> ens_D;
      ens_D.resize(ens_size_);
      for (int s = 0; s < ens_size_; s++)
      {
        ens_D[s] = Load_Ds_Data(s);
      }

      int N = ens_D[0]->Number_of_Vectors();
      std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> vecs;
      vecs.resize(N);
      for (int ell = 0; ell < N; ell++)
      {
        std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> vec_ell;
        vec_ell.resize(ens_size_);
        for (int s = 0; s < ens_size_; s++)
        {
          vec_ell[s] = (*ens_D[s])[ell];
        }
        vecs[ell] = HDSA::makePtr<HDSA::Ensemble_Vector<RealT>>(vec_ell);
      }
      HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(vecs);
      return D;
    }
  };

}

#endif
