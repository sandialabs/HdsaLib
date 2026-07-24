/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_COMM_HPP
#define HDSA_COMM_HPP

#include "Tpetra_Core.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_Array.hpp"
#include "HDSA_Ptr.hpp"

namespace HDSA
{

  template <class Ordinal>
  class Comm
  {
  private:
    HDSA::Ptr<const Teuchos::Comm<int>> comm_;

  public:
    Comm()
    {
      comm_ = Tpetra::getDefaultComm();
    }

    Comm(const HDSA::Ptr<const Teuchos::Comm<int>> &comm) : comm_(comm)
    {
    }

    Comm(HDSA::Ptr<Teuchos::MpiComm<int>> &comm) : comm_(comm)
    {
    }

    int getRank() const
    {
      return comm_->getRank();
    }

    int getSize() const
    {
      return comm_->getSize();
    }

    void barrier() const
    {
      comm_->barrier();
    }

    void broadcast(const int rootRank, const Ordinal bytes, char buffer[]) const
    {
      comm_->broadcast(rootRank, bytes, buffer);
    }

    int receive(const int sourceRank, const Ordinal bytes, char recvBuffer[]) const
    {
      int info = comm_->receive(sourceRank, bytes, recvBuffer);
      return info;
    }

    void send(const Ordinal bytes, const char sendBuffer[], const int destRank) const
    {
      comm_->send(bytes, sendBuffer, destRank);
    }

    void gatherAll(const Ordinal sendBytes, const char sendBuffer[], const Ordinal recvBytes, char recvBuffer[]) const
    {
      comm_->gatherAll(sendBytes, sendBuffer, recvBytes, recvBuffer);
    }

    HDSA::Ptr<HDSA::Comm<int>> createSubcommunicator(const std::vector<int> &ranks) const
    {
      Teuchos::Array<int> r;
      for (unsigned int k = 0; k < ranks.size(); k++)
      {
        r.push_back(ranks[k]);
      }
      HDSA::Ptr<Teuchos::Comm<int>> subcomm_teuchos = comm_->createSubcommunicator(r);
      HDSA::Ptr<HDSA::Comm<Ordinal>> subcomm = HDSA::makePtr<HDSA::Comm<Ordinal>>(subcomm_teuchos);
      return subcomm;
    }

    HDSA::Ptr<const Teuchos::Comm<int>> Get_Teuchos_Communicator() const
    {
      return comm_;
    }
  };

}

#endif
