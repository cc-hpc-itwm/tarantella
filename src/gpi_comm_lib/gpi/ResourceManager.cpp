#include "ResourceManager.hpp"

#include <algorithm>
#include <stdexcept>

namespace tarantella
{
  namespace GPI
  {
    ResourceManager& ResourceManager::get_instance(GPI::Context& context)
    {
      static auto instance = new ResourceManager(context);
      return *instance;
    }

    ResourceManager::ResourceManager(GPI::Context& context)
    : queueManager(GPI::QueueManager::get_instance()), 
      groupManager(), notificationManager(), segmentManager(context)
    { }

    void ResourceManager::make_segment_resources(GPI::SegmentID id, GPI::Group const& group, std::size_t size)
    {
      segmentManager.create_segment(id, group, size);
      notificationManager.register_segment(id);
    }

    GPI::Group const ResourceManager::make_group(std::vector<GPI::Rank> const& ranks)
    {
      return groupManager.create_group(ranks);
    }

    std::vector<GPI::Group> const& ResourceManager::get_groups() const
    {
      return groupManager.get_groups();
    }

    GPI::QueueID ResourceManager::get_queue_id_for_write_notify()
    {
      return queueManager.get_queue_id_for_write_notify();
    }

    void ResourceManager::wait_and_flush_queue()
    {
      queueManager.wait_and_flush_queue();
    }

    GPI::NotificationRange ResourceManager::get_notification_range(GPI::SegmentID id, std::size_t s)
    {
      return notificationManager.get_notification_range(id, s);
    }

    GPI::SegmentBuffer ResourceManager::get_buffer_of_size(GPI::SegmentID id, std::size_t s)
    {
      return segmentManager.get_buffer_of_size(id, s);
    }
  }
}