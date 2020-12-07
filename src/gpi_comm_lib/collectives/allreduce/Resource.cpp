#include "Resource.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      Resource::Resource(GPI::SegmentBuffer const& seg_buffer, 
                         GPI::NotificationManager::NotificationRange const& notification_range)
      : buffer(seg_buffer), range(notification_range)
      {}

      GPI::SegmentBuffer Resource::get_segment_buffer() const
      {
        return buffer;
      }

      GPI::NotificationManager::NotificationRange Resource::get_notification_range() const
      {
        return range;
      }
    }
  }
}