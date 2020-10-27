#include "GlobalContextFixture.hpp"
#include "gpi/Group.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <GASPI.h>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  BOOST_AUTO_TEST_CASE(gpigroup_allocate_one_group)
  {
    auto& context = GlobalContext::instance()->gpi_cont;
    auto nranks_in_group = context.get_comm_size() - 1;
    if (nranks_in_group > 0)
    {
      auto const group_ranks = gen_group_ranks(nranks_in_group);
      BOOST_REQUIRE_NO_THROW(GPI::Group const group(group_ranks));
    }
  }

  BOOST_AUTO_TEST_CASE(gpigroup_allocate_multiple_group_all)
  {
    auto &context = GlobalContext::instance()->gpi_cont;
    auto const group_ranks = gen_group_ranks(context.get_comm_size());
    GPI::Group const group1(group_ranks);
    GPI::Group const group2(group_ranks);
    GPI::Group const group3(group_ranks);

    BOOST_TEST_REQUIRE(group1.get_size() == group_ranks.size());
    BOOST_TEST_REQUIRE(group2.get_size() == group_ranks.size());
    BOOST_TEST_REQUIRE(group3.get_size() == group_ranks.size());
  }

  BOOST_AUTO_TEST_CASE(gpigroup_check_ranks_in_group)
  {
    auto &context = GlobalContext::instance()->gpi_cont;
    
    auto shuffled_ranks = gen_group_ranks(context.get_comm_size());
    std::shuffle(shuffled_ranks.begin(), shuffled_ranks.end(), std::mt19937(42));

    size_t const nranks_in_group = context.get_comm_size() / 2;
    if (nranks_in_group > 0)
    {
      auto group_ranks_list(shuffled_ranks);
      group_ranks_list.resize(nranks_in_group);
      GPI::Group const group(group_ranks_list);

      for (auto rank : shuffled_ranks)
      {
        auto const rank_iter = std::find(group_ranks_list.begin(), group_ranks_list.end(), rank);
        if (rank_iter != group_ranks_list.end()) // ranks in the `group_ranks_list` should be found in the group
        {
          BOOST_TEST_REQUIRE(group.contains_rank(rank));
        }
        else
        {
          BOOST_TEST_REQUIRE(!group.contains_rank(rank));
        }
      }
    }
  }

  BOOST_AUTO_TEST_CASE(gpigroup_throw_allocate_empty_group)
  {
    std::vector<gaspi_rank_t> group_ranks;
    BOOST_REQUIRE_THROW(GPI::Group const group(group_ranks), std::runtime_error);
  }

  BOOST_AUTO_TEST_CASE(gpigroup_multiple_overlapping_groups)
  {
    std::vector<std::unique_ptr<GPI::Group>> allocated_groups;
    auto& context = GlobalContext::instance()->gpi_cont;
    
    for (size_t nranks_in_group = 1; nranks_in_group <= context.get_comm_size(); ++nranks_in_group)
    {
      auto const group_ranks = gen_group_ranks(nranks_in_group);
      BOOST_REQUIRE_NO_THROW(allocated_groups.emplace_back(std::make_unique<GPI::Group>(group_ranks)));
      BOOST_TEST_REQUIRE(allocated_groups.back()->get_size() == nranks_in_group);

      if (context.get_rank() < nranks_in_group) // ranks lower than `nranks_in_group` should belong to the group
      {
        BOOST_TEST_REQUIRE(allocated_groups.back()->contains_rank(context.get_rank()));
      }
      else  // other ranks should not be part of the group
      {
        BOOST_TEST_REQUIRE(!allocated_groups.back()->contains_rank(context.get_rank()));
      }
    }
  }
}
