#include "broadcast.h"
#include "gpi/gaspiCheckReturn.hpp"
#include "mailBoxGaspi.h"

#include <vector>
#include <algorithm>

namespace tarantella
{
  namespace collectives
  {
    using tarantella::GPI::gaspiCheckReturn;
    
    broadcast::broadcast(
      const gaspi_rank_t master_,
      const long len,
      const gaspi_segment_id_t segment_,
      const gaspi_offset_t offset_,
      const gaspi_notification_id_t firstNotification_,
      queues& queues_ )
    :  totalLength(len),
       group(GASPI_GROUP_ALL),
       numRanks(getNumRanks()),
       rank(getRank()),
       masterRank(master_),
       segment(segment_),
       offset(offset_),
       firstNotification(firstNotification_),
       sender(queues_),
       status((rank == masterRank) ? 1 : numRanks){
    
      std::vector<gaspi_rank_t> ranks(numRanks);
      gaspiCheckReturn(gaspi_group_ranks(group, &ranks[0]),
                       "gaspi_group_ranks failed with");
      const unsigned long rankIndex = getRankIndex(rank, ranks);
    
      if (rank == masterRank) {
        setMaster(rankIndex, ranks);
      } else {
        setWorker(rankIndex, ranks);
      }
    }
    
    long broadcast::getNumRanks() const {
      gaspi_number_t size;
      gaspiCheckReturn(gaspi_group_size(group, &size),
                       "gaspi_group_size failed with ");
      return size;
    }
    
    long broadcast::getRank() {
      gaspi_rank_t rank;
      gaspiCheckReturn(gaspi_proc_rank(&rank),
                       "gaspi_proc_rank failed with ");
      return rank;
    }
    
    long broadcast::getRankIndex(gaspi_rank_t rank,
                                 const std::vector<gaspi_rank_t>& ranks) {
      unsigned long rankIndex;
      if (find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        throw std::runtime_error("rank not member of group");
      } else {
        rankIndex = find(ranks.begin(), ranks.end(), rank)
                  - ranks.begin();
      }
      return rankIndex;
    }
    
    void broadcast::setMaster(
      const unsigned long rankIndex,
      const std::vector<gaspi_rank_t>& ranks) {
      const gaspi_rank_t partner = ranks[getPartnerIndex(rankIndex)];
    
      receiver.push_back(&trigger);
    
      if (partner != rank) {
        for (long c=0; c < numRanks; c++) {
          writer::transferParameters job(
            true,
            partner,
            segment,
            offset + chunkIndexToByte(c),
            segment,
            offset + chunkIndexToByte(c),
            chunkIndexToByte(c + 1) - chunkIndexToByte(c),
            firstNotification + c);
          jobs.push_back(job);
        }
      }
    }
    
    inline unsigned long broadcast::getPartnerIndex(
      const unsigned long rankIndex) const {
      return (rankIndex + 1) % numRanks;
    }
    
    void broadcast::setWorker(
      const unsigned long rankIndex,
      const std::vector<gaspi_rank_t>& ranks) {
      const gaspi_rank_t partner = ranks[getPartnerIndex(rankIndex)];
    
      for (long c=0; c < numRanks; c++) {
        receiver.push_back(
          new mailBoxGaspi(segment, firstNotification + c));
    
        if (partner == masterRank) {
          jobs.push_back(writer::transferParameters());
        } else {
          writer::transferParameters transfer(
            true,
            partner,
            segment,
            offset + chunkIndexToByte(c),
            segment,
            offset + chunkIndexToByte(c),
            chunkIndexToByte(c + 1) - chunkIndexToByte(c),
            firstNotification + c);
          jobs.push_back(transfer);
        }
      }
    }
    
    inline unsigned long broadcast::chunkIndexToByte(
      const long chunkIndex) const {
      return ((totalLength * chunkIndex + numRanks - 1) / numRanks);
    }
    
    broadcast::~broadcast() {
      if (rank != masterRank) {
        for (unsigned long i=0; i < receiver.size(); i++) {
          delete receiver[i];
        }
      }
    }
    
    int broadcast::operator()() {
      const unsigned long phase = status.get();
      if (!receiver[phase]->gotNotification()) {
        return -1;
      }
    
      if (rank == masterRank) {
        for (unsigned long i=0; i < jobs.size(); i++) {
          sender(jobs[i]);
        }
      } else {
        sender(jobs[phase]);
      }
    
      return (status.increment() == 0) ? 0 : -1;
    }
    
    void broadcast::signal() {
      trigger.notify();
    }
    
    long broadcast::getNumberOfNotifications(const long numRanks) {
      return (numRanks > 1) ? numRanks : 0;
    }
    
    std::ostream& broadcast::report(std::ostream& s) const {
      const unsigned long phase = status.get();
      s << "total length: " << totalLength << std::endl
        << "numRanks: " << numRanks  << std::endl
        << "rank: " << rank << std::endl
        << "masterRank: " << masterRank << std::endl
        << "segment: " << long(segment) << std::endl
        << "offset: " << offset << std::endl
        << "firstNotification: " << firstNotification << std::endl
        << std::endl
        << "phase " << phase << std::endl;
      for (unsigned long i=0; i < jobs.size(); i++) {
        s << ".........................." << std::endl;
        s << "phase " << i << std::endl;
        if ((i==0) && (rank == masterRank)) {
          s << "Receiver: " << "user" << std::endl;
        } else {
          if (i < receiver.size()) {
            mailBoxGaspi* m = (mailBoxGaspi*) receiver[i];
            s << "Receiver: segment " << long(m->getSegmentID())
              << " notification ID " << m->getMailID() << std::endl;
          } else {
            s << "Receiver: idle" << std::endl;
          }
        }
    
        s << "Send    : ";
        jobs[i].report(s) << std::endl;
      }
      s << ".........................." << std::endl;
    
      return s;
    }
  }
}
    