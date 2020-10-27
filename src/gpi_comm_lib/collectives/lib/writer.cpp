#include "writer.h"
#include "gpi/gaspiCheckReturn.hpp"

#include <string>

namespace tarantella
{
  namespace collectives
  {
    const gaspi_size_t writer::MESSAGE_LENGTH_LIMIT = 0x40000000;

    using tarantella::GPI::gaspiCheckReturn;

    writer::transferParameters::transferParameters(
      bool a,
      gaspi_rank_t r,
      gaspi_segment_id_t sl,
      gaspi_offset_t ol,
      gaspi_segment_id_t sr,
      gaspi_offset_t orm,
      gaspi_size_t sz,
      gaspi_notification_id_t id)
    : active(a),
      rank(r),
      segmentLocal(sl),
      offsetLocal(ol),
      segmentRemote(sr),
      offsetRemote(orm),
      size(sz),
      notificationID(id)
    {}

    std::ostream& writer::transferParameters::report(std::ostream& s) const {
      if (active) {
        s << "rank " << rank
          << " | sl " << long(segmentLocal)
          << " ol " << offsetLocal
          << " | sr " << long(segmentRemote)
          << " or " << offsetRemote
          << " ID " << notificationID
          << " | sz " << size;
      } else {
        s << "idle";
      }
      return s;
    }

    writer::writer(queues& queues_)
    : queueSource(queues_) {}

    void writer::operator()(const transferParameters& p) {
      if (!p.active) return;
      //thread save? watch queue management!

      if (p.size > MESSAGE_LENGTH_LIMIT) {
        throw std::runtime_error("writer: message is too long");
      }

      gaspi_return_t err;
      gaspi_queue_id_t queueLocal = queueSource.get();
      while ((err = gaspi_write_notify(p.segmentLocal,
                                      p.offsetLocal,
                                      p.rank,
                                      p.segmentRemote,
                                      p.offsetRemote,
                                      p.size,
                                      p.notificationID,
                                      1,
                                      queueLocal,
                                      GASPI_BLOCK))
            != GASPI_SUCCESS) {
        if (err == GASPI_QUEUE_FULL) {
          queueLocal = queueSource.swap(queueLocal);
        } else {
          gaspiCheckReturn(err, "gaspi_write_notify failed with ");
        }
      }
    }
  }
}
