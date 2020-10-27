#pragma once

#include "Types.hpp"
#include "Group.hpp"

#include <memory>
#include <vector>

namespace tarantella
{
  namespace GPI
  {
    class GroupManager
    {
      public:
        GroupManager() = default;
        GroupManager(GroupManager const&) = delete;
        GroupManager& operator=(GroupManager const&) = delete;
        ~GroupManager() = default;

        GPI::Group const create_group(std::vector<GPI::Rank> const&);
        std::vector<GPI::Group> const& get_groups() const;

      private:
        std::vector<GPI::Group> groups;
    };
  }
}