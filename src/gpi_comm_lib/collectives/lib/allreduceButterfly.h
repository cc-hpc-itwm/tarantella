#pragma once

#include "allreduce.h"
#include "counter.h"
#include "gpi/Group.hpp"
#include "mailBox.h"
#include "mailBoxLocal.h"
#include "queues.h"
#include "reduce.h"
#include "writer.h"

#include <GASPI.h>

#include <iostream>

namespace tarantella
{
  namespace collectives
  {
    class nestedRingParameter {
    public:
      typedef unsigned long rankIndexType;
      typedef unsigned long levelType;
      typedef unsigned long bufferIndexType;
    
      nestedRingParameter(const rankIndexType numRanks_,
                          const rankIndexType rank_=0);
    
      rankIndexType getNumberOfRings() const;
      rankIndexType getRingLength(const levelType level) const;
      rankIndexType getLocalRankInRing(const levelType level) const;
      rankIndexType getGlobalRankToWriteInRing(const levelType level) const;
      bufferIndexType getBufferLength(const levelType level) const;
      bufferIndexType getBufferStart(const levelType level,
                                   const bufferIndexType buffer) const;
    
    private:
    
      typedef std::vector<unsigned long> ringIndicesType;
      typedef std::vector<unsigned long> ringSizesType;
      typedef std::vector<unsigned long> stridesType;
    
      static inline ringSizesType getRingSizes(rankIndexType numRanks);
      static inline stridesType getStrides(const ringSizesType& ringSizes);
      static inline ringIndicesType getRingIndices(const ringSizesType& ringSizes,
                                                   const rankIndexType rank);
    
      const rankIndexType numRanks;
      const rankIndexType rank;
      const ringSizesType ringSizes;
      const stridesType strides;
      const ringIndicesType ringIndices;
    };
    
    class allreduceButterfly : public allreduce {
    public:
    
      struct segmentBuffer {
        gaspi_segment_id_t segment;
        gaspi_offset_t offset;
        gaspi_notification_id_t firstNotification;
      };
    
      allreduceButterfly(const long len,
                        const dataType data,
                        const reductionType reduction,
                        const segmentBuffer segmentReduce,
                        const segmentBuffer segmentCommunicate,
                        queues& queues_,
                        GPI::Group const& group_);
      ~allreduceButterfly();
      int operator()();
      void signal();
    
      gaspi_pointer_t getReducePointer() const;
      static long getNumberOfElementsSegmentCommunicate(const long len,
                                                        const long numRanks);
      static unsigned long getNumberOfNotifications(const long numRanks);
      std::ostream& report(std::ostream& s) const;
    
    private:
    
      typedef nestedRingParameter::rankIndexType rankIndexType;
      typedef nestedRingParameter::bufferIndexType bufferIndexType;
      typedef std::pair<reduce::task, writer::transferParameters> jobType;
      
      inline long getNumRanks() const;
      static inline long getRank();
      std::vector<gaspi_rank_t> getRanks() const;
      static inline rankIndexType getRankIndex(
        gaspi_rank_t rank,
        const std::vector<gaspi_rank_t>& ranks);
      void setReduceScatter(const std::vector<gaspi_rank_t>& ranks);
      inline static char* getSegmentPointer(const gaspi_segment_id_t segment);
      inline unsigned long chunkIndexToByte(const long chunkIndex) const;
      void setAllToAll(const std::vector<gaspi_rank_t>& ranks);
    
      const long totalLength;
      const dataType dataElement;
      GPI::Group const group;
      const long numRanks;
      const gaspi_rank_t rank;
      const segmentBuffer locationReduce;
      const gaspi_pointer_t locationReducePointer;
      const segmentBuffer locationCommunicate;
    
      const nestedRingParameter topology;
    
      mailBoxLocal trigger;
      std::vector<mailBox *> receiver;
      std::vector<jobType> jobs;
    
      writer sender;
      reduce * reducer;
      counter status;
    };
  }
}
