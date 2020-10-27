#include "collectives/barrier/GPIBarrier.hpp"
#include "GlobalContextFixture.hpp"
#include "gpi/QueueManager.hpp"
#include "gpi/Segment.hpp"
#include "utilities.hpp"

#include <GASPI.h>

#include <boost/test/unit_test.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  namespace 
  {
    void check_queue_id_valid(GPI::QueueID qid)
    {
      gaspi_number_t max_num_queues_allowed;
      gaspi_queue_max(&max_num_queues_allowed);
      BOOST_TEST_REQUIRE(qid < max_num_queues_allowed);

      gaspi_number_t queue_max_size;
      gaspi_number_t queue_size;
      gaspi_queue_size_max(&queue_max_size);
      gaspi_queue_size(qid, &queue_size);
      BOOST_TEST_REQUIRE(queue_size + 2 <= queue_max_size);
    }

    void write_n_requests_to_neighbor(GPI::QueueManager &qmanager, 
                                      gaspi_number_t n_requests)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      std::size_t offset = 1;
      std::size_t buffer_size = 1;
      gaspi_notification_t notif_value = 1;
      GPI::Rank next_rank = (context.get_rank() + 1) % context.get_comm_size();

      GPI::Group group(gen_group_ranks(context.get_comm_size()));
      GPI::Segment segment(context, group, segment_id, size_in_bytes);

      collectives::Barrier::GPIBarrierAllRanks barrier;
      barrier.blocking_barrier();

      auto notif_range = std::make_pair<gaspi_notification_id_t, gaspi_notification_id_t>(0, n_requests);
      for (auto notif_id = notif_range.first; notif_id < notif_range.second; ++notif_id)
      {
        auto const qid = qmanager.get_queue_id_for_write_notify();
        check_queue_id_valid(qid);
        gaspi_write_notify(segment.get_id(), offset, next_rank,
                           segment.get_id(), offset, buffer_size,
                           notif_id, notif_value,
                           qid, GASPI_BLOCK);
      }

      for (auto i = 0UL; i < n_requests; ++i)
      {
        gaspi_notification_id_t notif_id;
        gaspi_notify_waitsome(segment.get_id(),
                              notif_range.first, notif_range.second - notif_range.first,
                              &notif_id, GASPI_BLOCK);
        gaspi_notify_reset(segment.get_id(), notif_id, &notif_value);
      }
    }
  }

   BOOST_AUTO_TEST_SUITE(queuemanager_unit)
   
    BOOST_AUTO_TEST_CASE(queuemanager_request_queue)
    {
      auto& qmanager = GPI::QueueManager::get_instance();

      auto const qid = qmanager.get_queue_id_for_write_notify();
      check_queue_id_valid(qid);
    }

    BOOST_AUTO_TEST_CASE(queuemanager_request_multiple_queues_without_notif)
    {
      auto& qmanager = GPI::QueueManager::get_instance();
      std::size_t nqueues = 100;

      for (auto i = 0UL; i < nqueues; ++i)
      {
        auto const qid = qmanager.get_queue_id_for_write_notify();
        check_queue_id_valid(qid);
      }
    }

    BOOST_AUTO_TEST_CASE(queuemanager_use_multiple_queues)
    {
      auto& qmanager = GPI::QueueManager::get_instance();

      gaspi_number_t max_queue_size;
      gaspi_queue_size_max(&max_queue_size);

      gaspi_number_t number_queues;
      gaspi_queue_num(&number_queues);

      auto const n_requests = 2 * max_queue_size / 2 * number_queues;
      write_n_requests_to_neighbor(qmanager, n_requests);
      qmanager.wait_and_flush_queue();

      // all queues should be empty
      gaspi_number_t num_queues;
      gaspi_queue_num(&num_queues);
      for (auto qid = 0UL; qid < num_queues; ++qid)
      {
        gaspi_number_t queue_size;
        gaspi_queue_size(qid, &queue_size);
        BOOST_TEST_REQUIRE(queue_size == 0);
      }
    }

  BOOST_AUTO_TEST_SUITE_END()
}
