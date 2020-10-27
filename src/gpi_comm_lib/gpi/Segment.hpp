#pragma once

#include "Context.hpp"
#include "Group.hpp"

#include <GASPI.h>

#include <cstddef>

namespace tarantella
{
  namespace GPI
  {
    class Segment
    {
      public:

        Segment(Context& context, Group const&, SegmentID, std::size_t );
        Segment(Segment const& other) = delete;
        Segment& operator=(Segment const& other) = delete;
        Segment(Segment&& other) = delete;
        Segment& operator=(Segment&& other) = delete;
        ~Segment();

        std::size_t get_size() const;
        SegmentID get_id() const;
        void* get_ptr() const;

      private:

        Context& context;
        Group const group;

        SegmentID const id;
        std::size_t const size;
        void* /* const */ ptr;
    };
  }
}

