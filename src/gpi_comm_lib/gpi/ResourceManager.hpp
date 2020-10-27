#pragma once

#include "gpi/Context.hpp"
#include "gpi/GroupManager.hpp"
#include "gpi/NotificationManager.hpp"
#include "gpi/QueueManager.hpp"
#include "gpi/SegmentManager.hpp"
#include "gpi/SegmentBuffer.hpp"
#include "gpi/Types.hpp"

#include <GASPI.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    class ResourceManager
    {
      public:
        static ResourceManager &get_instance(GPI::Context &);
        ResourceManager() = delete;
        ResourceManager(ResourceManager const&) = delete;
        ResourceManager& operator=(ResourceManager const&) = delete;
        ~ResourceManager() = default;

        void make_segment_resources(GPI::SegmentID, GPI::Group const&, std::size_t);
        GPI::Group const make_group(std::vector<GPI::Rank> const&);
        std::vector<GPI::Group> const& get_groups() const;
        GPI::QueueID get_queue_id_for_write_notify();
        void wait_and_flush_queue();
        GPI::NotificationRange get_notification_range(GPI::SegmentID, std::size_t);
        GPI::SegmentBuffer get_buffer_of_size(GPI::SegmentID, std::size_t);

      private:
        ResourceManager(GPI::Context&);
  
        GPI::QueueManager& queueManager;
        GPI::GroupManager groupManager;
        GPI::NotificationManager notificationManager;
        GPI::SegmentManager segmentManager;
    };
  }
}
