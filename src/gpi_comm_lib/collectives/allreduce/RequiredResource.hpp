#pragma once

#include "Types.hpp"

#include <cstddef>
#include <unordered_map>

namespace tarantella
{
  namespace collectives
  {
    namespace Allreduce
    {
      class RequiredResource
      {
        public:
          RequiredResource();

          void set_buffer_size_bytes(std::size_t);
          void set_num_notifications(std::size_t);

          std::size_t get_buffer_size_bytes() const;
          std::size_t get_num_notifications() const;

        private:
          std::size_t buffer_size_bytes;
          std::size_t num_notifications;
      };

      using RequiredResourceList = std::unordered_map<SegmentType, RequiredResource>;
    }
  }
}