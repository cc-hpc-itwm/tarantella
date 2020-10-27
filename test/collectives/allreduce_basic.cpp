
#include "AllreduceTestSetupGenerator.hpp"
#include "GlobalContextFixture.hpp"
#include "allreduceButterfly.h"

#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

using boost::test_tools::per_element;

struct AllreduceBasicTestCase
{
  unsigned long nelems;
  unsigned long nprocs;
  unsigned long expected_size_comm_seg;
  unsigned long expected_nnotifs;
};
namespace std
{
  std::ostream& operator<<(std::ostream& os, AllreduceBasicTestCase const& test)
  {
    os << "Nelems=" << test.nelems << ", Nprocs=" << test.nprocs;
    os << std::endl;
    return os;
  }
}

namespace tarantella
{
    std::vector<AllreduceBasicTestCase> test_cases
    {
      // nelems, nprocs, expected_size_comm_buf, expected_nnotifs
      {       1,      1,                      0,                0},
      {       5,      1,                      0,                0},
      {       1,      2,                      1,                1},
      {       2,      2,                      1,                1},
      {       7,      2,                      4,                1},
      {       1,      4,                      3,                2},
      {       4,      4,                      3,                2},
    };

    BOOST_AUTO_TEST_SUITE(allreduce_basic_unit)

    BOOST_DATA_TEST_CASE(allreduce_size_segm_comm, test_cases, test_case)
    {
      auto nelems_buffer = test_case.nelems;
      auto nprocs = test_case.nprocs;

      auto nelems_segment_comm = collectives::allreduceButterfly::getNumberOfElementsSegmentCommunicate(
                                                    nelems_buffer, nprocs) ;
      BOOST_TEST_REQUIRE(nelems_segment_comm == test_case.expected_size_comm_seg);
    }

    BOOST_DATA_TEST_CASE(allreduce_nnotifications, test_cases, test_case)
    {
      auto nprocs = test_case.nprocs;
      auto nnotifications = collectives::allreduceButterfly::getNumberOfNotifications(nprocs);
      BOOST_TEST_REQUIRE(nnotifications == test_case.expected_nnotifs);
    }
    BOOST_AUTO_TEST_SUITE_END()
}
