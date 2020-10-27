#pragma once

#include "gpi/Context.hpp"
#include "gpi/ResourceManager.hpp"

#include <numeric>

namespace tarantella
{
  namespace distribution
  {
    class DataParallelGroupBuilder
    {
      public:
        DataParallelGroupBuilder(GPI::Context& context)
        : context(context)
        { }

        GPI::Group const get_group()
        {
          auto& resource_manager = context.get_resource_manager();
          auto const num_ranks = context.get_comm_size();

          std::vector<GPI::Rank> all_ranks(num_ranks);
          std::iota(all_ranks.begin(), all_ranks.end(), static_cast<GPI::Rank>(0));

          return resource_manager.make_group(all_ranks);
        }

      private:
        GPI::Context& context;
    };
  }
}