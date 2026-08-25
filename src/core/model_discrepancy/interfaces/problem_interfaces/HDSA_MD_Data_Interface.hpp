/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_HPP
#define HDSA_MD_DATA_INTERFACE_HPP

#include "HDSA_Vector.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Transient_Vector.hpp"
#include "HDSA_Transient_Vector_Const.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Data_Interface
  {

  private:
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt_;
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> D_;
    HDSA::Ptr<HDSA::Vector<RealT>> data_shift_;
    bool is_opt_data_loaded_, is_hifi_data_loaded_;

  public:
    MD_Data_Interface()
    {
      is_opt_data_loaded_ = false;
      is_hifi_data_loaded_ = false;
    }

    virtual ~MD_Data_Interface()
    {
    }

    void Load_Data(void)
    {
      Load_Opt_Data();
      Load_HiFi_Data();
    }

    void Load_Opt_Data(void)
    {
      u_opt_ = Load_Optimal_u();
      z_opt_ = Load_Optimal_z();
      data_shift_ = u_opt_->Clone();
      data_shift_->Zeros();
      is_opt_data_loaded_ = true;
    }

    void Load_HiFi_Data(void)
    {
      Z_ = Load_Z_Data();
      D_ = Load_D_Data();
      is_hifi_data_loaded_ = true;
    }

    void Center_Data()
    {
      data_shift_->Set_Scalar(1.0);
      RealT val = data_shift_->Dot(*(*D_)[0]) / static_cast<RealT>(data_shift_->Dimension());
      data_shift_->Set_Scalar(val);
      for (int k = 0; k < D_->Number_of_Vectors(); k++)
      {
        (*D_)[k]->Scaled_Plus(-1.0, *data_shift_);
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u(void) const = 0;

    virtual HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z(void) const = 0;

    virtual HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data(void) const = 0;

    virtual HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data(void) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Set_Z_and_D(
        const HDSA::Ptr<HDSA::MultiVector<RealT>> &Z,
        const HDSA::Ptr<HDSA::MultiVector<RealT>> &D)
    {
      HDSA_TEST_FOR_EXCEPTION(Z == HDSA::nullPtr, std::logic_error,
                              "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                              "Input Z multivector is null." << std::endl);

      HDSA_TEST_FOR_EXCEPTION(D == HDSA::nullPtr, std::logic_error,
                              "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                              "Input D multivector is null." << std::endl);

      HDSA_TEST_FOR_EXCEPTION(Z->Number_of_Vectors() != D->Number_of_Vectors(), std::logic_error,
                              "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                              "Z and D must contain the same number of vectors." << std::endl);

      if (!is_opt_data_loaded_)
      {
        Load_Opt_Data();
      }

      const int num_vecs = Z->Number_of_Vectors();

      for (int k = 0; k < num_vecs; ++k)
      {
        HDSA_TEST_FOR_EXCEPTION((*Z)[k] == HDSA::nullPtr, std::logic_error,
                                "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                                "Encountered null vector in Z." << std::endl);

        HDSA_TEST_FOR_EXCEPTION((*D)[k] == HDSA::nullPtr, std::logic_error,
                                "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                                "Encountered null vector in D." << std::endl);

        HDSA_TEST_FOR_EXCEPTION((*Z)[k]->Dimension() != z_opt_->Dimension(), std::logic_error,
                                "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                                "A Z vector has dimension inconsistent with z_opt." << std::endl);

        HDSA_TEST_FOR_EXCEPTION((*D)[k]->Dimension() != u_opt_->Dimension(), std::logic_error,
                                "Error in HDSA::MD_Data_Interface::Set_Z_and_D: "
                                "A D vector has dimension inconsistent with u_opt." << std::endl);
      }

      Z_ = Z;
      D_ = D;
      is_hifi_data_loaded_ = true;
    }

    virtual HDSA::Ptr<HDSA::MultiVector<RealT>> Read_Spatial_Node_Data() const
    {
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error: HDSA::MD_Data_Interface::Read_Spatial_Node_Data is a virtual function that was called but never implemented" << std::endl);
      HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_nodes;
      return spatial_nodes;
    }

    virtual HDSA::Ptr<const HDSA::Vector<RealT>> Extract_State_Component(const HDSA::Vector<RealT> &u, int component_id) const
    {
      (void) component_id;
      HDSA::Ptr<const HDSA::Vector<RealT>> u_component = HDSA::makePtrFromRef(u);
      return u_component;
    }

    virtual void Set_State_Component(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &u_component, int component_id) const
    {
      (void) component_id;
      u.Set(u_component);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // functions to manage abstraction for stationary and transient problems
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<const HDSA::Vector<RealT>> Extract_State_Component(const HDSA::Vector<RealT> &u, int component_id, bool check_transient) const
    {
      HDSA::Ptr<const HDSA::Vector<RealT>> u_component;
      if (check_transient)
      {
        if (const HDSA::Transient_Vector<RealT> *u_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&(u)))
        {
          int n_t = u_trans->Get_n_t();
          std::vector<HDSA::Ptr<const HDSA::Vector<RealT>>> u_component_trans;
          u_component_trans.resize(n_t);
          for (int k = 0; k < n_t; k++)
          {
            u_component_trans[k] = Extract_State_Component(*(*u_trans)[k], component_id);
          }
          u_component = HDSA::makePtr<HDSA::Transient_Vector_Const<RealT>>(u_component_trans);
        }
        else
        {
          u_component = Extract_State_Component(u, component_id);
        }
      }
      else
      {
        u_component = Extract_State_Component(u, component_id);
      }
      return u_component;
    }

    void Set_State_Component(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &u_component, int component_id, bool check_transient) const
    {
      if (check_transient)
      {
        if (HDSA::Transient_Vector<RealT> *u_trans = dynamic_cast<HDSA::Transient_Vector<RealT> *>(&(u)))
        {
          const HDSA::Transient_Vector<RealT> *u_component_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&(u_component));
          int n_t = u_trans->Get_n_t();
          for (int k = 0; k < n_t; k++)
          {
            Set_State_Component(*(*u_trans)[k], *(*u_component_trans)[k], component_id);
          }
        }
        else
        {
          Set_State_Component(u, u_component, component_id);
        }
      }
      else
      {
        Set_State_Component(u, u_component, component_id);
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // accessor functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<const HDSA::Vector<RealT>> Get_u_opt(void)
    {
      if (!is_opt_data_loaded_)
      {
        Load_Opt_Data();
      }
      return u_opt_;
    }

    HDSA::Ptr<const HDSA::Vector<RealT>> Get_z_opt(void)
    {
      if (!is_opt_data_loaded_)
      {
        Load_Opt_Data();
      }
      return z_opt_;
    }

    HDSA::Ptr<const HDSA::MultiVector<RealT>> Get_Z(void)
    {
      if (!is_hifi_data_loaded_)
      {
        Load_HiFi_Data();
      }
      return Z_;
    }

    HDSA::Ptr<const HDSA::MultiVector<RealT>> Get_D(void)
    {
      if (!is_hifi_data_loaded_)
      {
        Load_HiFi_Data();
      }
      return D_;
    }

    HDSA::Ptr<const HDSA::Vector<RealT>> Get_data_shift(void)
    {
      if (!is_opt_data_loaded_)
      {
        Load_Opt_Data();
      }
      return data_shift_;
    }
  };

}

#endif
