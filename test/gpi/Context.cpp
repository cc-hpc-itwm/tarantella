#include "collectives/barrier/GPIBarrier.hpp"
#include "GlobalContextFixture.hpp"
#include "gpi/gaspiCheckReturn.hpp"
#include "utilities.hpp"

#include <numeric>
#include <random>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

namespace tarantella
{
  BOOST_GLOBAL_FIXTURE( GlobalContext );
  namespace
  {
    std::size_t get_num_allocated_segments()
    {
      gaspi_number_t allocated_segments_num;
      tarantella::GPI::gaspiCheckReturn(gaspi_segment_num(&allocated_segments_num),
                                        "get number of segments");
      return allocated_segments_num;
    }
  }

  BOOST_AUTO_TEST_CASE(gpicontext_comm_size)
  {
    BOOST_TEST(GlobalContext::instance()->gpi_cont.get_comm_size() > 0);
  }

  BOOST_AUTO_TEST_CASE(gpicontext_allocate_segment)
  {
    auto &context = GlobalContext::instance()->gpi_cont;
    GPI::Group group(gen_group_ranks(context.get_comm_size()));
    GPI::SegmentID segment_id = 0;
    std::size_t size_in_bytes = 1000;
    
    BOOST_REQUIRE_NO_THROW(context.allocate_segment(segment_id, group, size_in_bytes));
    collectives::Barrier::GPIBarrierAllRanks barrier;
    barrier.blocking_barrier();

    BOOST_TEST_REQUIRE(get_num_allocated_segments() == 1);

    BOOST_REQUIRE_NO_THROW(context.deallocate_segment(segment_id, group));
    BOOST_TEST_REQUIRE(get_num_allocated_segments() == 0);
  }
}
