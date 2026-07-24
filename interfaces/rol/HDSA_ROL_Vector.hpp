/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_ROL_VECTOR_HPP
#define HDSA_ROL_VECTOR_HPP

#include "ROL_Vector.hpp"
#include "ROL_StdVector.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Random_Number_Generator.hpp"

namespace HDSA
{

  template <class RealT>
  class ROL_Vector : public HDSA::Vector<RealT>
  {

  public:
    ROL::Ptr<ROL::Vector<RealT>> rol_vec;
    static ROL::Elementwise::NormalRandom<RealT> nr;
    HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator;
    HDSA::Ptr<const HDSA::Comm<int>> comm;

    ROL_Vector(ROL::Ptr<ROL::Vector<RealT>> &rol_vec_in, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator_in, const HDSA::Ptr<const HDSA::Comm<int>> &comm_in) : rol_vec(rol_vec_in), random_number_generator(random_number_generator_in), comm(comm_in)
    {
    }

    ROL_Vector(ROL::Vector<RealT> &rol_vec_in, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator_in, const HDSA::Ptr<const HDSA::Comm<int>> &comm_in) : random_number_generator(random_number_generator_in), comm(comm_in)
    {
      rol_vec = rol_vec_in.clone();
      rol_vec->set(rol_vec_in);
    }

    ROL_Vector(const ROL::Ptr<ROL::Vector<RealT>> &rol_vec_in, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator_in, const HDSA::Ptr<const HDSA::Comm<int>> &comm_in) : rol_vec(rol_vec_in), random_number_generator(random_number_generator_in), comm(comm_in) {};

    virtual ~ROL_Vector()
    {
    }

    // Clone the vector
    HDSA::Ptr<HDSA::Vector<RealT>> Clone() const override
    {
      ROL::Ptr<ROL::Vector<RealT>> rol_vec_clone = rol_vec->clone();
      rol_vec_clone->zero(); // ROL clone() vector is not initialized
      return Teuchos::rcp(new HDSA::ROL_Vector<RealT>(rol_vec_clone,random_number_generator,comm));
    }

    // compute the Dot product of this and x
    RealT Dot(const HDSA::Vector<RealT> &x) const override
    {
      const HDSA::ROL_Vector<RealT> &ex = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(x);
      RealT val = ex.rol_vec->dot(*rol_vec);
      return val;
    }

    // add alpha*x to this
    void Scaled_Plus(const RealT alpha, const HDSA::Vector<RealT> &x) override
    {
      const HDSA::ROL_Vector<RealT> &ex = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(x);
      rol_vec->axpy(alpha, *ex.rol_vec);
    }

    // return vector Dimension
    int Dimension() const override
    {
      return rol_vec->dimension();
    }

    // Set this=val elementwise
    void Set_Scalar(const RealT val) override
    {
      rol_vec->setScalar(val);
    }

    void Randomize_Standard_Normal() override
    {
      rol_vec->applyUnary(nr);
    }

    void Write_to_File(const std::string &name) const override
    {
      try
      {
        ROL::Ptr<std::vector<RealT>> vec = dynamic_cast<ROL::StdVector<RealT> &>(*rol_vec).getVector();
        std::ofstream fout;
        fout.open(name);
        for (int i = 0; i < rol_vec->dimension(); i++)
        {
          fout << std::setprecision(16) << (*vec)[i] << "  ";
        }
        fout.close();
      }
      catch (...)
      {
        std::cout << "Write_to_File is currently not supported for this vector type" << std::endl;
      }
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Generate_Std_Vector(int r) const override
    {
      HDSA::Ptr<HDSA::Vector<RealT>> vec = HDSA::makePtr<HDSA::Std_Vector<RealT>>(r, random_number_generator, comm);
      return vec;
    }
  };

  template <>
  ROL::Elementwise::NormalRandom<double> ROL_Vector<double>::nr = ROL::Elementwise::NormalRandom<double>(0.0, 1.0);

}

#endif
