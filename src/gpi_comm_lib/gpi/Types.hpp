#pragma once

#include <cstddef>
#include <utility>

#include <GASPI.h>

namespace tarantella
{
  namespace GPI
  {
    using Rank = short unsigned int;
    using SegmentID = unsigned char;
    using GroupID = unsigned long;

    using NotificationID = std::size_t;
    using NotificationRange = std::pair<NotificationID, NotificationID>;
    using QueueID = gaspi_queue_id_t;
  }
}