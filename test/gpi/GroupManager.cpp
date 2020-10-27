#include "GlobalContextFixture.hpp"
#include "gpi/GroupManager.hpp"
#include "utilities.hpp"

#include <boost/test/unit_test.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  BOOST_AUTO_TEST_SUITE(groupmanager_unit)
    BOOST_AUTO_TEST_CASE(groupmanager_no_predefined_group)
    {
      GPI::GroupManager gmanager;

      auto const& groups = gmanager.get_groups();
      BOOST_TEST_REQUIRE(groups.size() == 0);
    }

    BOOST_AUTO_TEST_CASE(groupmanager_create_group)
    {
      auto& context = GlobalContext::instance()->gpi_cont;
      GPI::GroupManager gmanager;

      auto const group = gmanager.create_group(gen_group_ranks(context.get_comm_size()));
      
      BOOST_REQUIRE_NO_THROW(gmanager.get_groups());
    }

    BOOST_AUTO_TEST_CASE(groupmanager_create_empty_group)
    {
      GPI::GroupManager gmanager;
      BOOST_REQUIRE_THROW(gmanager.create_group({}), std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(groupmanager_create_multiple_groups)
    {
      auto& context = GlobalContext::instance()->gpi_cont;
      GPI::GroupManager gmanager;

      for (auto group_size = 1UL; group_size <= context.get_comm_size(); ++group_size)
      {
        // create group regardless of whether it contains the current rank or not
        BOOST_REQUIRE_NO_THROW(gmanager.create_group(gen_group_ranks(group_size)));
        auto const&groups = gmanager.get_groups();

        if (context.get_rank() < group_size)  // groups contain consecutive ranks between [0, group_size)
        {
          BOOST_TEST_REQUIRE(groups.back().contains_rank(context.get_rank()));
        }
        else
        {
          BOOST_TEST_REQUIRE(!groups.back().contains_rank(context.get_rank()));
        }
        
      }
      auto num_created_groups = context.get_comm_size();
      BOOST_TEST_REQUIRE(num_created_groups == gmanager.get_groups().size());
    }
  BOOST_AUTO_TEST_SUITE_END()
}
