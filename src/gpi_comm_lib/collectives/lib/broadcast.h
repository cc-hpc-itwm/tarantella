#pragma once

#include "writer.h"
#include "mailBox.h"
#include "mailBoxLocal.h"
#include "counter.h"
#include "queues.h"

#include <GASPI.h>
#include <iostream>

namespace tarantella
{
  namespace collectives
  {
    class broadcast {
    public:
      broadcast(const gaspi_rank_t master_,
                const long len,
                const gaspi_segment_id_t segment_,
                const gaspi_offset_t offset_,
                const gaspi_notification_id_t firstNotification_,
                queues& queues_);
      ~broadcast();
      int operator()();
      void signal();
      static long getNumberOfNotifications(const long numRanks);
      std::ostream& report(std::ostream& s) const;
    
    private:
    
      long getNumRanks() const;
      static long getRank();
      static long getRankIndex(gaspi_rank_t rank,
                               const std::vector<gaspi_rank_t>& ranks);
      void setMaster(const unsigned long rankIndex,
                     const std::vector<gaspi_rank_t>& ranks);
      inline unsigned long getPartnerIndex(const unsigned long rankIndex) const;
      void setWorker(const unsigned long rankIndex,
                     const std::vector<gaspi_rank_t>& ranks);
      inline unsigned long chunkIndexToByte(const long chunkIndex) const;
      inline static char* getSegmentPointer(const gaspi_segment_id_t segment);
    
      const long totalLength;
      const gaspi_group_t group;
      const long numRanks;
      const gaspi_rank_t rank;
      const gaspi_rank_t masterRank;
      const gaspi_segment_id_t segment;
      const gaspi_offset_t offset;
      const gaspi_notification_id_t firstNotification;
    
      mailBoxLocal trigger;
      std::vector<mailBox *> receiver;
      std::vector<writer::transferParameters> jobs;
    
      writer sender;
      counter status;
    };
  }
}
    