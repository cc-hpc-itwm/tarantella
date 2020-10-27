#include "allreduceButterfly.h"
#include "gpi/gaspiCheckReturn.hpp"
#include "mailBoxGaspi.h"
#include "gpi/Group.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace tarantella
{
  namespace collectives
  {
    using tarantella::GPI::gaspiCheckReturn;
  
    nestedRingParameter::nestedRingParameter(const rankIndexType numRanks_,
                                             const rankIndexType rank_) :
      numRanks(numRanks_),
      rank(rank_),
      ringSizes(getRingSizes(numRanks)),
      strides(getStrides(ringSizes)),
      ringIndices(getRingIndices(ringSizes, rank_)) {}
  
    inline nestedRingParameter::ringSizesType nestedRingParameter::getRingSizes(
      rankIndexType numRanks) {
      ringSizesType s;
  
      unsigned long limit = std::sqrt(numRanks) + 2;
  
      for (unsigned long factor=2; factor < limit; factor++) {
        while ((numRanks % factor) == 0) {
          s.push_back(factor);
          numRanks /= factor;
        }
      }
  
      if (numRanks > 1) {
        s.push_back(numRanks);
      }
  
      return s;
    }
  
    inline nestedRingParameter::stridesType nestedRingParameter::getStrides(
      const ringSizesType& ringSizes) {
      const long numLevels = ringSizes.size();
      stridesType s(ringSizes.size());
      unsigned long factor = 1;
      for (long level=numLevels - 1; level >= 0; level--) {
        s[level] = factor;
        factor *= ringSizes[level];
      }
  
      return s;
    }
  
    inline nestedRingParameter::ringIndicesType
    nestedRingParameter::getRingIndices(const ringSizesType& ringSizes,
                                        const rankIndexType rank) {
      ringIndicesType indices;
  
      rankIndexType product = 1;
      for (unsigned long i=0; i < ringSizes.size(); i++) {
        indices.push_back((rank / product) % ringSizes[i]);
        product *= ringSizes[i];
      }
  
      return indices;
    }
  
    nestedRingParameter::rankIndexType
    nestedRingParameter::getNumberOfRings() const{
      return ringSizes.size();
    }
  
    nestedRingParameter::rankIndexType nestedRingParameter::getRingLength(
      const levelType level) const {
      return ringSizes[level];
    }
  
    nestedRingParameter::rankIndexType nestedRingParameter::getLocalRankInRing(
      const levelType level) const {
      return ringIndices[level];
    }
  
    nestedRingParameter::rankIndexType
    nestedRingParameter::getGlobalRankToWriteInRing(
      const levelType level) const {
      long numLevels = ringSizes.size();
      rankIndexType r = 0;
      for (long i=numLevels - 1; i > long(level); i--) {
        r = ringIndices[i] + ringSizes[i] * r;
      }
      const rankIndexType next = (ringIndices[level] + 1) % ringSizes[level];
      r =  next + ringSizes[level] * r;
      for (long i=long(level) - 1; i >= 0; i--) {
        r = ringIndices[i] + ringSizes[i] * r;
      }
      return r;
    }
  
    nestedRingParameter::bufferIndexType nestedRingParameter::getBufferLength(
      const levelType level) const {
      return strides[level];
    }
  
    nestedRingParameter::bufferIndexType nestedRingParameter::getBufferStart(
      const levelType level,
      const bufferIndexType buffer) const {
      // we assume that each global rank aggregates on each level the buffer
      // that matches the local ring id. This buffer is
      // I.E. getBufferStart(level, getRankInRing(level))
      // -> getBufferStart(level, getRankInRing(level)) + getBufferLength(level)
  
      bufferIndexType s = 0;
      for (unsigned long i=0; i < level; i++) {
        s += ringIndices[i] * strides[i];
      }
      s += buffer * strides[level];
  
      return s;
    }
  
    allreduceButterfly::allreduceButterfly(
      const long len,
      const dataType data,
      const reductionType reduction,
      const segmentBuffer locationReduce_,
      const segmentBuffer locationCommunicate_,
      queues& queues_,
      GPI::Group const& group_
      )
    :  totalLength(len),
       dataElement(data),
       group(group_),
       numRanks(getNumRanks()),
       rank(getRank()),
       locationReduce(locationReduce_),
       locationReducePointer(getSegmentPointer(locationReduce_.segment)
                             + locationReduce_.offset),
       locationCommunicate(locationCommunicate_),
       topology(numRanks, getRankIndex(rank, getRanks())),
       sender(queues_),
       reducer(getReduce(data, reduction)),
       status(2 * getNumberOfNotifications(numRanks) + 1){
  
      std::vector<gaspi_rank_t> ranks = getRanks();
  
      setReduceScatter(ranks);
      setAllToAll(ranks);
    }
  
    long allreduceButterfly::getNumRanks() const {
      return group.get_size();
    }
  
    long allreduceButterfly::getRank() {
      gaspi_rank_t rank;
      gaspiCheckReturn(gaspi_proc_rank(&rank),
                       "gaspi_proc_rank failed with ");
      return rank;
    }
  
    std::vector<gaspi_rank_t> allreduceButterfly::getRanks() const {
      return group.get_ranks();
    }
  
    unsigned long allreduceButterfly::getRankIndex(
      gaspi_rank_t rank,
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
  
    void allreduceButterfly::setReduceScatter(
      const std::vector<gaspi_rank_t>& ranks) {
      gaspi_notification_id_t nextNotification
        = locationCommunicate.firstNotification;
      gaspi_offset_t nextLocalCommunicationBufferByte = 0;
      const char* const reductionSourceBasePointer =
        getSegmentPointer(locationCommunicate.segment)
        + locationCommunicate.offset;
      char* const reductionDestinationBasePointer =
        getSegmentPointer(locationReduce.segment)
        + locationReduce.offset;
  
      receiver.push_back(&trigger);
      jobs.push_back(jobType());
  
      for (unsigned long ring=0; ring < topology.getNumberOfRings(); ring++) {
  
        const rankIndexType ringLength = topology.getRingLength(ring);
        const rankIndexType ringRank = topology.getLocalRankInRing(ring);
        const bufferIndexType bufferLengthIndex = topology.getBufferLength(ring);
        const gaspi_rank_t outgoingGlobalRank =
          ranks[topology.getGlobalRankToWriteInRing(ring)];
        gaspi_offset_t nextRemoteCommunicationBufferByte
          = nextLocalCommunicationBufferByte;
  
  
        for (unsigned long loop=0; loop < ringLength - 1; loop++) {
          const unsigned long currentJob = receiver.size() - 1;
          receiver.push_back(
            new mailBoxGaspi(locationCommunicate.segment, nextNotification));
          jobs.push_back(jobType());
  
          const bufferIndexType sendBufferID =
            (ringRank + ringLength - loop - 1) % ringLength;
          const bufferIndexType sendStartIndex =
            topology.getBufferStart(ring, sendBufferID);
          const gaspi_offset_t sendStartByte =
            chunkIndexToByte(sendStartIndex);
          const long sendLengthByte =
            chunkIndexToByte(sendStartIndex + bufferLengthIndex)
            - sendStartByte;
          const writer::transferParameters transfer(
            true,
            outgoingGlobalRank,
            locationReduce.segment,
            locationReduce.offset + sendStartByte,
            locationCommunicate.segment,
            locationCommunicate.offset + nextRemoteCommunicationBufferByte,
            sendLengthByte,
            nextNotification);
          jobs[currentJob].second = transfer;
  
          const bufferIndexType receiveBufferID =
            (ringRank + ringLength - loop - 2) % ringLength;
          const bufferIndexType receiveStartIndex =
            topology.getBufferStart(ring, receiveBufferID);
          const gaspi_offset_t receiveStartByte =
            chunkIndexToByte(receiveStartIndex);
          const long receiveLengthByte =
            chunkIndexToByte(receiveStartIndex + bufferLengthIndex)
            - receiveStartByte;
          const reduce::task copy(
            reductionSourceBasePointer + nextLocalCommunicationBufferByte,
            reductionDestinationBasePointer + receiveStartByte,
            receiveLengthByte / getDataTypeSize(dataElement));
          jobs[currentJob + 1].first = copy;
  
          nextNotification++;
          nextRemoteCommunicationBufferByte += sendLengthByte;
          nextLocalCommunicationBufferByte += receiveLengthByte;
        }
      }
  
      jobs.back().first.scaling = numRanks;
    }
  
    inline char* allreduceButterfly::getSegmentPointer(
      const gaspi_segment_id_t segment) {
      gaspi_pointer_t p;
      gaspiCheckReturn(gaspi_segment_ptr(segment, &p),
                       "failed getting segment pointer");
      return (char*) p;
    }
  
    inline unsigned long allreduceButterfly::chunkIndexToByte(
      const long chunkIndex) const {
      return ((totalLength * chunkIndex + numRanks - 1) / numRanks)
             * getDataTypeSize(dataElement);
    }
  
    void allreduceButterfly::setAllToAll(
      const std::vector<gaspi_rank_t>& ranks) {
      gaspi_notification_id_t nextNotification = locationReduce.firstNotification;
  
      for (long ring=topology.getNumberOfRings() - 1; ring >=0 ; ring--) {
  
        const rankIndexType ringLength = topology.getRingLength(ring);
        const rankIndexType ringRank = topology.getLocalRankInRing(ring);
        const bufferIndexType bufferLengthIndex = topology.getBufferLength(ring);
        const gaspi_rank_t outgoingGlobalRank =
          ranks[topology.getGlobalRankToWriteInRing(ring)];
  
        for (unsigned long loop=0; loop < ringLength - 1; loop++) {
          const unsigned long currentJob = receiver.size() - 1;
          receiver.push_back(
            new mailBoxGaspi(locationReduce.segment, nextNotification));
          jobs.push_back(jobType());
  
          const bufferIndexType transferBufferID =
            (ringRank + ringLength - loop) % ringLength;
          const bufferIndexType transferStartIndex =
            topology.getBufferStart(ring, transferBufferID);
          const gaspi_offset_t transferStartByte =
            chunkIndexToByte(transferStartIndex);
          const long transferLengthByte =
            chunkIndexToByte(transferStartIndex + bufferLengthIndex)
            - transferStartByte;
  
          const writer::transferParameters transfer(
            true,
            outgoingGlobalRank,
            locationReduce.segment,
            locationReduce.offset + transferStartByte,
            locationReduce.segment,
            locationReduce.offset + transferStartByte,
            transferLengthByte,
            nextNotification);
          jobs[currentJob].second = transfer;
  
          nextNotification++;
        }
      }
    }
  
    allreduceButterfly::~allreduceButterfly() {
      delete reducer;
      for (unsigned long i=1; i < receiver.size(); i++) {
        delete receiver[i];
      }
    }
  
    int allreduceButterfly::operator()() {
      const unsigned long phase = status.get();
      // could be a problem if we overtake one iteration?
      if (!receiver[phase]->gotNotification()) {
        return -1;
      }
  
      reducer->operator()(jobs[phase].first);
      // hier schon freigeben?
      sender(jobs[phase].second);
  
      return (status.increment() == 0) ? 0 : -1;
    }
  
    void allreduceButterfly::signal() {
      trigger.notify();
    }
  
    gaspi_pointer_t allreduceButterfly::getReducePointer() const {
      return locationReducePointer;
    }
  
    long allreduceButterfly::getNumberOfElementsSegmentCommunicate(
      const long len,
      const long numRanks) {
      return ((len + numRanks - 1) / numRanks) * (numRanks - 1);
    }
  
    unsigned long allreduceButterfly::getNumberOfNotifications(
      const long numRanks) {
      const nestedRingParameter topology(numRanks);
  
      gaspi_notification_id_t notifications = 0;
      for (unsigned long i=0; i < topology.getNumberOfRings(); i++) {
        notifications += topology.getRingLength(i) - 1;
      }
  
      return notifications;
    }
  
    std::ostream& allreduceButterfly::report(std::ostream& s) const {
      char* pr = getSegmentPointer(locationReduce.segment);
      char* pc = getSegmentPointer(locationCommunicate.segment);
      const unsigned long phase = status.get();
      s << "total length: " << totalLength << std::endl
        << "dataElement: " << dataElement << std::endl
        << "numRanks: " << numRanks  << std::endl
        << "rank: " << rank << std::endl
        << "topology.getNumberOfRings" << topology.getNumberOfRings() << std::endl
        << "getNumberOfNotifications(): "
        << getNumberOfNotifications(numRanks) << std::endl
        << "segmentReduce: " << long(locationReduce.segment) << std::endl
        << "offsetReduce: " << locationReduce.offset << std::endl
        << "firstNotificationReduce: " << locationReduce.firstNotification
        << std::endl
        << "segmentCommunicate: " << long(locationCommunicate.segment)
        << std::endl
        << "offsetCommunicate: " << locationCommunicate.offset << std::endl
        << "firstNotificationCommunicate: "
        << locationCommunicate.firstNotification << std::endl
        << "pointer segment reduce     : "
        << (void*)getSegmentPointer(locationReduce.segment) << std::endl
        << "pointer segment communicate: "
        << (void*)getSegmentPointer(locationCommunicate.segment) << std::endl
        << "phase " << phase << std::endl;
      for (unsigned long i=0; i < jobs.size(); i++) {
        s << ".........................." << std::endl;
        s << "phase " << i << std::endl;
        if (i==0) {
          s << "Receiver: " << "user" << std::endl;
        } else {
          mailBoxGaspi* m = (mailBoxGaspi*) receiver[i];
          s << "Receiver: segment " << long(m->getSegmentID())
            << " notification ID " << m->getMailID() << std::endl;
        }
  
        if (jobs[i].first.len > 0) {
          s << "Reduce  : src " << jobs[i].first.source
            << " (" << (char*)jobs[i].first.source - pc << ")"
            << " dst " << jobs[i].first.destination
            << " (" << (char*)jobs[i].first.destination - pr << ")"
            << " ele " << jobs[i].first.len
            << " (" << jobs[i].first.len * getDataTypeSize(dataElement) << ")"
            << std::endl;
        } else {
          s << "Reduce  : idle" << std::endl;
        }
  
        s << "Send    : ";
        jobs[i].second.report(s) << std::endl;
      }
      s << ".........................." << std::endl;
  
      return s;
    }
  }
}
