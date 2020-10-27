#pragma once

#include "Types.hpp"

#include <GASPI.h>

#include <cstddef>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    class Group
    {
      public:
        Group(std::vector<Rank> const&);

        std::size_t get_size() const;
        bool contains_rank(Rank) const;
        std::vector<Rank> const& get_ranks() const;

      private:
        std::vector<Rank> ranks;
    };
  }
}
