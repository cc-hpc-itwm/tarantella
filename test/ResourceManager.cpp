#include "GlobalContextFixture.hpp"
#include "gpi/ResourceManager.hpp"
#include "utilities.hpp"

#include <boost/test/unit_test.hpp>

namespace std
{
  std::ostream& operator<< (std::ostream& os, tarantella::GPI::ResourceManager const&)
  {
    return os;
  }
}

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  BOOST_AUTO_TEST_SUITE(resourcemanager_unit)
    BOOST_AUTO_TEST_CASE(resourcemanager_require_queue)
    {
      auto& context = GlobalContext::instance()->gpi_cont;
      BOOST_REQUIRE_NO_THROW(context.get_resource_manager().get_queue_id_for_write_notify());
    }

    BOOST_AUTO_TEST_CASE(resourcemanager_require_group)
    {
      auto& context = GlobalContext::instance()->gpi_cont;
      auto& resource_manager = context.get_resource_manager();
      BOOST_REQUIRE_NO_THROW(resource_manager.make_group(gen_group_ranks(context.get_comm_size())));
    }

    BOOST_AUTO_TEST_CASE(resourcemanager_require_notification)
    {
      auto& resource_manager = GlobalContext::instance()->gpi_cont.get_resource_manager();
      GPI::SegmentID segment_id = 1;
      auto const num_ranks = GlobalContext::instance()->gpi_cont.get_comm_size();
      auto group_all = resource_manager.make_group(gen_group_ranks(num_ranks));
      
      std::size_t segment_size = 10;
      BOOST_REQUIRE_NO_THROW(resource_manager.make_segment_resources(segment_id, group_all, segment_size));
      BOOST_REQUIRE_NO_THROW(resource_manager.get_notification_range(segment_id, 2));
    }

    BOOST_AUTO_TEST_CASE(resourcemanager_require_segment_buffer)
    {
      auto& resource_manager = GlobalContext::instance()->gpi_cont.get_resource_manager();
      GPI::SegmentID segment_id = 2;
      auto const num_ranks = GlobalContext::instance()->gpi_cont.get_comm_size();
      auto group_all = resource_manager.make_group(gen_group_ranks(num_ranks));

      std::size_t segment_size = 10;
      std::size_t buffer_size = 10;
      BOOST_REQUIRE_NO_THROW(resource_manager.make_segment_resources(segment_id, group_all, segment_size));
      BOOST_REQUIRE_NO_THROW(resource_manager.get_buffer_of_size(segment_id, buffer_size));
    }
  BOOST_AUTO_TEST_SUITE_END()
}
