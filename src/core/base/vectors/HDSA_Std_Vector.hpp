/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_STDVECTOR_HPP
#define HDSA_STDVECTOR_HPP

#include "HDSA_Vector.hpp"
#include "HDSA_Random_Number_Generator.hpp"

namespace HDSA
{

  template <class RealT>
  class Std_Vector : public HDSA::Vector<RealT>
  {

  private:
    int dim_;
    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
    HDSA::Ptr<std::vector<RealT>> vec_;
    const HDSA::Ptr<const HDSA::Comm<int>> comm_;

  public:
    Std_Vector(int dim, const HDSA::Ptr<const HDSA::Comm<int>> &comm) : dim_(dim), random_number_generator_(HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>()), comm_(comm)
    {
      vec_ = HDSA::makePtr<std::vector<RealT>>(dim, 0.0);
    }

    Std_Vector(int dim, int seed, const HDSA::Ptr<const HDSA::Comm<int>> &comm) : dim_(dim), random_number_generator_(HDSA::Random_Number_Generator<RealT>(seed)), comm_(comm)
    {
      vec_ = HDSA::makePtr<std::vector<RealT>>(dim, 0.0);
    }

    Std_Vector(int dim, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, const HDSA::Ptr<const HDSA::Comm<int>> &comm) : dim_(dim), random_number_generator_(random_number_generator), comm_(comm)
    {
      vec_ = HDSA::makePtr<std::vector<RealT>>(dim, 0.0);
    }

    Std_Vector(std::vector<RealT> &vec_in, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, const HDSA::Ptr<const HDSA::Comm<int>> &comm) : dim_(vec_in.size()), random_number_generator_(random_number_generator), comm_(comm)
    {
      vec_ = HDSA::makePtr<std::vector<RealT>>(vec_in.size());
      for (int k = 0; k < vec_in.size(); k++)
      {
        (*vec_)[k] = vec_in[k];
      }
    }

    ~Std_Vector()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Overloading pure virtual functions in HDSA::Vector base class
    //////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<HDSA::Vector<RealT>> Clone() const override
    {
      HDSA::Ptr<HDSA::Vector<RealT>> vec = HDSA::makePtr<Std_Vector<RealT>>(dim_, random_number_generator_, comm_);
      return vec;
    }

    // compute the Dot product of this and x
    RealT Dot(const HDSA::Vector<RealT> &x) const override
    {
      RealT val = 0.0;
      const Std_Vector<RealT> x_std = dynamic_cast<const Std_Vector<RealT> &>(x);
      for (int k = 0; k < dim_; k++)
      {
        val += (*vec_)[k] * (x_std(k));
      }
      return val;
    }

    // add alpha*x to this
    void Scaled_Plus(const RealT alpha, const HDSA::Vector<RealT> &x) override
    {
      const Std_Vector<RealT> x_std = dynamic_cast<const Std_Vector<RealT> &>(x);
      for (int k = 0; k < dim_; k++)
      {
        (*vec_)[k] += alpha * (x_std(k));
      }
    }

    // return vector Dimension
    int Dimension() const override
    {
      return dim_;
    }

    // Set this=val elementwise
    void Set_Scalar(const RealT val) override
    {
      for (int k = 0; k < dim_; k++)
      {
        (*vec_)[k] = val;
      }
    }

    void Randomize_Standard_Normal() override
    {
      for (int k = 0; k < dim_; k++)
      {
        (*vec_)[k] = random_number_generator_->Generate_Standard_Normal_Sample();
      }
      if (comm_->getSize() > 1)
      {
        comm_->barrier();
        char *buff = (char *)(&(*vec_)[0]);
        comm_->broadcast(0, 8 * dim_, buff);
      }
    }

    void Write_to_File(const std::string &name) const override
    {
      std::ofstream fout;
      fout.open(name);
      for (int i = 0; i < dim_; i++)
      {
        fout << std::setprecision(16) << (*vec_)[i] << "  ";
      }
      fout.close();
    }

    virtual RealT Get_Entry(int k) const override
    {
      RealT val = (*vec_)[k];
      return val;
    }

    virtual void Set_Entry(int k, RealT val) override
    {
      (*vec_)[k] = val;
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Generate_Std_Vector(int r) const override
    {
      HDSA::Ptr<HDSA::Vector<RealT>> vec = HDSA::makePtr<HDSA::Std_Vector<RealT>>(r, random_number_generator_, comm_);
      return vec;
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Function specific to this class for convenience
    //////////////////////////////////////////////////////////////////////////////////

    // Access underlying std::vector
    const HDSA::Ptr<std::vector<RealT>> get_std_vec(void) const
    {
      return vec_;
    }

    RealT operator()(int k) const
    {
      return (*vec_)[k];
    }

  };

}
#endif
