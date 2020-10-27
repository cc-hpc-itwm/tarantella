#include "Group.hpp"

#include <GASPI.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    Group::Group(std::vector<Rank> const &ranks_to_add)
    : ranks(ranks_to_add)
    {
      if (ranks.size() == 0)
      {
        throw std::runtime_error("Group: Cannot create empty group");
      }
      std::sort(ranks.begin(), ranks.end());
    }

    std::size_t Group::get_size() const
    {
      return ranks.size();
    }

    bool Group::contains_rank(Rank rank) const
    {
      auto const iter = std::find(ranks.begin(), ranks.end(), rank);
      return iter != ranks.end();
    }

    std::vector<Rank> const& Group::get_ranks() const
    {
      return ranks;
    }
  }
}
