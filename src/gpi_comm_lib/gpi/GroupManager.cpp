#include "GroupManager.hpp"

namespace tarantella
{
  namespace GPI
  {
    GPI::Group const GroupManager::create_group(std::vector<GPI::Rank> const& ranks)
    {
      groups.emplace_back(ranks);
      return groups.back();
    }

    std::vector<GPI::Group> const& GroupManager::get_groups() const
    { 
      return groups;
    }
  }
}