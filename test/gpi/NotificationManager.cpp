#include "GlobalContextFixture.hpp"
#include "gpi/NotificationManager.hpp"

#include <GASPI.h>

#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE(GlobalContext);

  BOOST_AUTO_TEST_SUITE(notificationmanager_unit)

    BOOST_AUTO_TEST_CASE(notificationmanager_simple_range)
    {
      GPI::NotificationManager notif_manager;
      GPI::SegmentID segment_id = 0;
      notif_manager.register_segment(segment_id);

      std::size_t const num_notifications(10);
      auto notification_range = notif_manager.get_notification_range(segment_id, num_notifications);
      BOOST_TEST_REQUIRE(notification_range.first == 0);
      BOOST_TEST_REQUIRE(notification_range.second == num_notifications);
    }

    BOOST_AUTO_TEST_CASE(notificationmanager_throw_max_range)
    {
      GPI::NotificationManager notif_manager;
      GPI::SegmentID segment_id = 0;
      notif_manager.register_segment(segment_id);

      gaspi_number_t max_num_notifications;
      gaspi_notification_num(&max_num_notifications);

      BOOST_REQUIRE_THROW(notif_manager.get_notification_range(segment_id, max_num_notifications + 1),
                          std::runtime_error);

      BOOST_REQUIRE_NO_THROW(notif_manager.get_notification_range(segment_id, max_num_notifications));
    }

    BOOST_AUTO_TEST_CASE(notificationmanager_allow_empty_range)
    {
      GPI::NotificationManager notif_manager;
      GPI::SegmentID segment_id = 0;
      notif_manager.register_segment(segment_id);

      std::size_t num_notifs = 0;
      auto notification_range = notif_manager.get_notification_range(segment_id, num_notifs);
      BOOST_TEST_REQUIRE(notification_range.first == notification_range.second);
    }

    BOOST_AUTO_TEST_CASE(notificationmanager_consecutive_ranges)
    {
      GPI::NotificationManager notif_manager;
      GPI::SegmentID segment_id = 0;
      notif_manager.register_segment(segment_id);

      std::vector<std::size_t> const notification_range_sizes{1, 10, 20, 100, 3};
      std::size_t previous_max_notification(0);
      GPI::NotificationManager::NotificationRange notification_range;
      for (auto num_notifs : notification_range_sizes)
      {
        notification_range = notif_manager.get_notification_range(segment_id, num_notifs);
        BOOST_TEST_REQUIRE(notification_range.first == previous_max_notification);
        BOOST_TEST_REQUIRE(notification_range.second == previous_max_notification + num_notifs);

        previous_max_notification += num_notifs;
      }

      std::size_t total_num_notifs = std::accumulate(notification_range_sizes.begin(), 
                                                     notification_range_sizes.end(), 0);
      BOOST_TEST_REQUIRE(notification_range.second == total_num_notifs);
    }

    BOOST_AUTO_TEST_CASE(notificationmanager_unregistered_segment)
    {
      GPI::NotificationManager notif_manager;
      GPI::SegmentID segment_id = 1;
      std::size_t num_notifs = 5;

      BOOST_REQUIRE_THROW(notif_manager.get_notification_range(segment_id, num_notifs),
                          std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(notificationmanager_multiple_segments)
    {
      GPI::NotificationManager notif_manager;
      std::vector<GPI::SegmentID> segment_ids{1,2,3,4,5};
      std::size_t num_notifs = 5;

      for (auto const segment_id: segment_ids)
      {
        notif_manager.register_segment(segment_id);
      }
      for (auto const segment_id: segment_ids)
      {
        BOOST_REQUIRE_NO_THROW(notif_manager.get_notification_range(segment_id, num_notifs));
      }
    }

  BOOST_AUTO_TEST_SUITE_END()
}
