#include "RequiredResource.hpp"

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      RequiredResource::RequiredResource()
      : buffer_size_bytes(0), num_notifications(0)
      {}

      void RequiredResource::set_buffer_size_bytes(std::size_t size_bytes)
      {
        buffer_size_bytes = size_bytes;
      }

      void RequiredResource::set_num_notifications(std::size_t notifications)
      {
        num_notifications = notifications;
      }

      std::size_t RequiredResource::get_buffer_size_bytes() const
      {
        return buffer_size_bytes;
      }

      std::size_t RequiredResource::get_num_notifications() const
      {
        return num_notifications;
      }
    }
  }
}