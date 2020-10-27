#include "collectives/barrier/GPIBarrier.hpp"
#include "GlobalContextFixture.hpp"
#include "gpi/gaspiCheckReturn.hpp"
#include "gpi/SegmentManager.hpp"
#include "utilities.hpp"

#include <GASPI.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <boost/test/unit_test.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );

  namespace
  {
    void create_segment_with_id(GPI::Context& context, GPI::SegmentManager& segmentmanager, 
                                GPI::SegmentID segment_id, std::size_t size_in_bytes)
    {
      GPI::Group group(gen_group_ranks(context.get_comm_size()));

      segmentmanager.create_segment(segment_id, group, size_in_bytes);
      collectives::Barrier::GPIBarrierAllRanks barrier;
      barrier.blocking_barrier();
    }
  }

  BOOST_AUTO_TEST_SUITE(segmentmanager_unit)
    BOOST_AUTO_TEST_CASE(segmentmanager_create_manager)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      BOOST_REQUIRE_NO_THROW(GPI::SegmentManager segmentmanager(context));
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_create_segment)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);
      BOOST_REQUIRE_NO_THROW(create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes));
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_create_empty_segment)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      GPI::Group group(gen_group_ranks(context.get_comm_size()));
      std::size_t size_zero = 0;
      
      GPI::SegmentManager segmentmanager(context);
      BOOST_REQUIRE_THROW(segmentmanager.create_segment(segment_id, group, size_zero),
                          std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_create_multiple_segments)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      std::vector<GPI::SegmentID> segment_ids{1,5,6,31};
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);

      for (auto segment_id : segment_ids)
      {
        BOOST_REQUIRE_NO_THROW(create_segment_with_id(context, segmentmanager,
                                                      segment_id, size_in_bytes));
      }

      gaspi_number_t allocated_segments_num;
      GPI::gaspiCheckReturn(gaspi_segment_num(&allocated_segments_num),
                       "get number of segments");
      BOOST_TEST_REQUIRE(allocated_segments_num == segment_ids.size());
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_duplicate_segment_id)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 5;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);

      BOOST_REQUIRE_NO_THROW(create_segment_with_id(context, segmentmanager,
                                                    segment_id, size_in_bytes));
      BOOST_REQUIRE_THROW(create_segment_with_id(context, segmentmanager,
                                                 segment_id, size_in_bytes),
                          std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_delete_manager)
    {
      gaspi_number_t initially_allocated_segments_num;
      GPI::gaspiCheckReturn(gaspi_segment_num(&initially_allocated_segments_num),
                       "get number of segments");

      auto &context = GlobalContext::instance()->gpi_cont;
      {
        GPI::SegmentID segment_id = 0;
        std::size_t size_in_bytes = 1000;
        GPI::SegmentManager segmentmanager(context);
        create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);
      }

      // segment manager should be out of scope and all segments deallocated
      gaspi_number_t allocated_segments_num;
      GPI::gaspiCheckReturn(gaspi_segment_num(&allocated_segments_num),
                       "get number of segments");
      BOOST_TEST_REQUIRE(allocated_segments_num == initially_allocated_segments_num);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_create_segment_buffer)
    {
      auto &context = GlobalContext::instance()->gpi_cont;

      GPI::SegmentID segment_id = 0;
      GPI::SegmentManager segmentmanager(context);
      std::size_t const size_in_bytes = 64;
      create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);

      std::size_t const expected_first_offset = 0;
      auto const segment_buffer = segmentmanager.get_buffer_of_size(segment_id, size_in_bytes);
      BOOST_TEST_REQUIRE(segment_buffer.get_size() == size_in_bytes);
      BOOST_TEST_REQUIRE(segment_buffer.get_offset() == expected_first_offset);
      
      auto const buffer_pointer = reinterpret_cast<std::byte*>(
                                  context.get_segment_pointer(segment_buffer.get_segment_id()));
      BOOST_TEST_REQUIRE(reinterpret_cast<std::byte*>(segment_buffer.get_ptr()) == buffer_pointer);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_segment_buffer_empty)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);
      create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);

      std::size_t const needed_buffer_size_in_bytes = 0;
      BOOST_REQUIRE_NO_THROW(segmentmanager.get_buffer_of_size(segment_id, needed_buffer_size_in_bytes));
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_segment_buffer_max_size)
    {
      auto &context = GlobalContext::instance()->gpi_cont;

      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);
      create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);

      BOOST_REQUIRE_NO_THROW(segmentmanager.get_buffer_of_size(segment_id, size_in_bytes));
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_segment_buffer_too_large)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);
      create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);

      BOOST_REQUIRE_THROW(segmentmanager.get_buffer_of_size(segment_id, size_in_bytes + 1),
                          std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_segment_buffer_beyond_max_size)
    {
      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      std::size_t size_in_bytes = 1000;
      GPI::SegmentManager segmentmanager(context);
      create_segment_with_id(context, segmentmanager, segment_id, size_in_bytes);

      segmentmanager.get_buffer_of_size(segment_id, size_in_bytes - 1);
      BOOST_REQUIRE_THROW(segmentmanager.get_buffer_of_size(segment_id, size_in_bytes),
                          std::runtime_error);
    }

    BOOST_AUTO_TEST_CASE(segmentmanager_multiple_segment_buffers)
    {
      std::vector<std::size_t> const sizes_in_bytes {10, 3, 56, 100, 1};
      std::size_t total_segment_size_in_bytes = std::accumulate(sizes_in_bytes.begin(), 
                                                                sizes_in_bytes.end(), 0);

      auto &context = GlobalContext::instance()->gpi_cont;
      GPI::SegmentID segment_id = 0;
      GPI::SegmentManager segmentmanager(context);
      create_segment_with_id(context, segmentmanager, segment_id, total_segment_size_in_bytes);

      std::size_t current_offset = 0;
      for (auto size_in_bytes : sizes_in_bytes)
      {
        auto const segment_buffer = segmentmanager.get_buffer_of_size(segment_id, size_in_bytes);
        BOOST_TEST_REQUIRE(segment_buffer.get_size() == size_in_bytes);
        BOOST_TEST_REQUIRE(segment_buffer.get_offset() == current_offset);
        BOOST_TEST_REQUIRE(segment_buffer.get_segment_id() == segment_id);

        auto const buffer_pointer = reinterpret_cast<std::byte*>(
                                    context.get_segment_pointer(segment_buffer.get_segment_id())) +
                                    current_offset;
        BOOST_TEST_REQUIRE(reinterpret_cast<std::byte*>(segment_buffer.get_ptr()) == buffer_pointer);

        current_offset += size_in_bytes;
      } 
    }

  BOOST_AUTO_TEST_SUITE_END()
}
