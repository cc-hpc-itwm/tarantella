#pragma once

#include "queues.h"

#include <GASPI.h>
#include <ostream>

namespace tarantella
{
  namespace collectives
  {
  class writer {
  public:
    struct transferParameters {
      bool active;
      gaspi_rank_t rank;
      gaspi_segment_id_t segmentLocal;
      gaspi_offset_t offsetLocal;
      gaspi_segment_id_t segmentRemote;
      gaspi_offset_t offsetRemote;
      gaspi_size_t size;
      gaspi_notification_id_t notificationID;
      transferParameters(
        bool a = false,
        gaspi_rank_t r = 0,
        gaspi_segment_id_t sl = 0,
        gaspi_offset_t ol = 0,
        gaspi_segment_id_t sr = 0,
        gaspi_offset_t orm = 0,
        gaspi_size_t sz = 0,
        gaspi_notification_id_t id = 0);
      std::ostream& report(std::ostream& s) const;
    };

    writer(queues& queues_);
    void operator()(const transferParameters& p);

  private:

    static const gaspi_size_t MESSAGE_LENGTH_LIMIT;

    queues& queueSource;
  };

  }
}