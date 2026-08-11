/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_TPETRA_VECTOR_HPP
#define HDSA_TPETRA_VECTOR_HPP

#include "HDSA_Vector.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Map.hpp"
#include "MatrixMarket_Tpetra.hpp"

namespace HDSA
{

  template <class RealT,
            class LO = Tpetra::Map<>::local_ordinal_type,
            class GO = Tpetra::Map<>::global_ordinal_type,
            class Node = Tpetra::Map<>::node_type>
  class Tpetra_Vector : public HDSA::Vector<RealT>
  {

  private:
    const HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> tpetra_vec_;
    const HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> map_;
    const HDSA::Ptr<const Teuchos::Comm<int>> comm_;
    const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;

  public:
    Tpetra_Vector(const HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> &tpetra_vec, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator)
        : tpetra_vec_(tpetra_vec), map_(tpetra_vec_->getMap()), comm_(map_->getComm()),
          random_number_generator_(random_number_generator) {}

    ~Tpetra_Vector()
    {
    }

    HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> getVector() const
    {
      return tpetra_vec_;
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Clone() const override
    {
      int n = tpetra_vec_->getNumVectors();
      return HDSA::makePtr<Tpetra_Vector>(HDSA::makePtr<Tpetra::MultiVector<RealT, LO, GO, Node>>(map_, n), random_number_generator_);
    }

    RealT Dot(const HDSA::Vector<RealT> &x) const override
    {
      const Tpetra_Vector &ex = dynamic_cast<const Tpetra_Vector &>(x);
      int n = tpetra_vec_->getNumVectors();
      Teuchos::Array<RealT> val(n, 0);
      tpetra_vec_->dot(*ex.getVector(), val.view(0, n));
      RealT xy(0);
      for (int i = 0; i < n; ++i)
      {
        xy += val[i];
      }
      return xy;
    }

    void Scaled_Plus(const RealT alpha, const HDSA::Vector<RealT> &x) override
    {
      RealT one(1);
      const Tpetra_Vector &ex = dynamic_cast<const Tpetra_Vector &>(x);
      tpetra_vec_->update(alpha, *ex.getVector(), one);
    }

    int Dimension() const override
    {
      int nVecs = static_cast<int>(tpetra_vec_->getNumVectors());
      int dim = static_cast<int>(tpetra_vec_->getGlobalLength());
      return nVecs * dim;
    }

    void Set_Scalar(const RealT val) override
    {
      tpetra_vec_->putScalar(static_cast<double>(val));
    }

    void Randomize_Standard_Normal() override
    {
      int nVecs = static_cast<int>(tpetra_vec_->getNumVectors());
      for (int j = 0; j < nVecs; j++)
      {
        auto vecT_data = tpetra_vec_->getDataNonConst(j);
        for (int i = 0; i < tpetra_vec_->getLocalLength(); i++)
        {
          vecT_data[i] = random_number_generator_->Generate_Standard_Normal_Sample();
        }
      }
    }

    void Write_to_File(const std::string &name) const override
    {
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<>> vecWriter;
      vecWriter.writeDenseFile(name, tpetra_vec_);
      // std::string mapfile = "map_" + name;
      // vecWriter.writeMapFile(mapfile, *(tpetra_vec_->getMap()));
    }

    RealT Get_Entry(int k) const override
    {
      bool isOwned = map_->isNodeGlobalElement(k);
      RealT val = 0.0;
      if (isOwned)
      {
        int localIndex = map_->getLocalElement(k);
        Teuchos::ArrayRCP<const RealT> vecT_data = tpetra_vec_->get1dView();
        val = vecT_data[localIndex];
      }
      return val;
    }

    void Set_Entry(int k, RealT val) override
    {
      bool isOwned = map_->isNodeGlobalElement(k);
      if (isOwned)
      {
        tpetra_vec_->replaceGlobalValue(k, 0, val);
      }
      comm_->barrier();
    }

  };

}
#endif
