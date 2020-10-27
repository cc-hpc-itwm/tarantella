#pragma once

#include "Context.hpp"

#include <cstddef>
#include <unordered_map>
#include <utility>

namespace tarantella
{
  namespace GPI
  {
    class NotificationManager
    {
      public:
        using NotificationID = std::size_t;
        using NotificationRange = std::pair<NotificationID, NotificationID>;

        NotificationManager();
        NotificationManager(NotificationManager const&) = delete;
        NotificationManager& operator=(NotificationManager const &) = delete;
        ~NotificationManager() = default;

        void register_segment(GPI::SegmentID);
        NotificationRange get_notification_range(GPI::SegmentID, std::size_t);

      private:
        std::size_t const max_notification_id;
        std::unordered_map<GPI::SegmentID, NotificationID> next_notification_ids;
    };
  } 
} 