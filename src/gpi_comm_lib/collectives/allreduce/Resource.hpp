#pragma once

#include "gpi/NotificationManager.hpp"
#include "gpi/SegmentBuffer.hpp"
#include "Types.hpp"

#include <cstddef>
#include <unordered_map>

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      class Resource
      {
        public:
          Resource(GPI::SegmentBuffer const&, 
                   GPI::NotificationManager::NotificationRange const&);

          GPI::SegmentBuffer get_segment_buffer() const;
          GPI::NotificationManager::NotificationRange get_notification_range() const;
        
        private:
          const GPI::SegmentBuffer buffer;
          const GPI::NotificationManager::NotificationRange range;
      };

      using ResourceList = std::unordered_map<SegmentType, Resource>;
    }
  }
}