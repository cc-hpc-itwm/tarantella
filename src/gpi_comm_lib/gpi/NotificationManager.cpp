#include "gpi/NotificationManager.hpp"
#include "gaspiCheckReturn.hpp"

#include <GASPI.h>

#include <stdexcept>

namespace tarantella
{
  namespace GPI
  {
    namespace 
    {
      std::size_t get_number_available_notifications()
      {
        gaspi_number_t notifications_available;
        gaspiCheckReturn(gaspi_notification_num(&notifications_available),
                        "[NotificationManager::get_number_available_notifications()] GASPI:\
                          Could not get number of available notifications");
        return notifications_available;
      }
    }

    NotificationManager::NotificationManager()
    : max_notification_id(get_number_available_notifications()), next_notification_ids()
    { }

    void NotificationManager::register_segment(GPI::SegmentID id)
    {
      if(next_notification_ids.find(id) != next_notification_ids.end())
      {
        throw std::runtime_error("[NotificationManager::register_segment]:\
                                  Segment already registered");
      }
      next_notification_ids[id] = 0UL;
    }

    NotificationManager::NotificationRange
     NotificationManager::get_notification_range(GPI::SegmentID id, std::size_t size)
    {
      if(next_notification_ids.find(id) == next_notification_ids.end())
      {
        throw std::runtime_error("[NotificationManager::get_notification_range]:\
                                  Segment not registered");
      }

      if(next_notification_ids[id] + size > max_notification_id)
      {
        throw std::runtime_error("[NotificationManager::get_notification_range]:\
                                  Not enough notifications left");
      }

      NotificationManager::NotificationRange const range = {next_notification_ids[id],
                                                            next_notification_ids[id] + size};
      next_notification_ids[id] += size;
      return range;
    }
  }
}