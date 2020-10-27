#pragma once

#include "Types.hpp"

#include <GASPI.h>

#include <cstddef>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    class Group;
    class ResourceManager;

    class Context
    {
      public:

        Context();
        Context(Context const& other) = delete;
        Context& operator=(Context const& other) = delete;
        ~Context();

        Rank get_rank() const;
        std::size_t get_comm_size() const;
        tarantella::GPI::ResourceManager& get_resource_manager();

        void allocate_segment(SegmentID id, Group const&, std::size_t total_size);
        void deallocate_segment(SegmentID id, Group const&);
        void* get_segment_pointer(SegmentID id) const;

      private:
        Rank rank;
        std::size_t comm_size;
        size_t const timeout_millis = 1000;
    };
  }
}
