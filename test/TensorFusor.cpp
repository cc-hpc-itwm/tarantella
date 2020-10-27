#include "FusedTensorInfo.hpp"
#include "utilities.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

namespace tarantella
{
  namespace collectives
  {
    std::vector<std::vector<TensorInfo>> non_fusion_test_cases
    {
      { 
        // test case #1
        // (tensor_id, num_elements, element_type)
        {1, 8, BufferElementType::FLOAT}
      },
      { 
        // test case #2
        {5, 4 * 1000, BufferElementType::FLOAT}
      },
      { 
        // test case #3
        {5, 10, BufferElementType::FLOAT},
        {9, 10, BufferElementType::FLOAT}
      },
      {
        // test case #4
        {1, 8, BufferElementType::FLOAT},
        {2, 8, BufferElementType::FLOAT},
        {3, 8, BufferElementType::FLOAT},
        {4, 8, BufferElementType::FLOAT},
        {5, 8, BufferElementType::FLOAT},
        {6, 9, BufferElementType::FLOAT},
      },
      {
        // test case #5
        {9, 3, BufferElementType::FLOAT},
        {1, 8, BufferElementType::FLOAT},
        {0, 9, BufferElementType::FLOAT},
        {4, 1, BufferElementType::FLOAT},
        {5, 17, BufferElementType::FLOAT},
        {6, 5, BufferElementType::FLOAT},
      },
    };

    std::vector<TensorInfo> fusion_test_case
    {
      {0, 94, BufferElementType::FLOAT},
      {1, 17, BufferElementType::FLOAT},
      {2, 2, BufferElementType::FLOAT},
      {3, 81, BufferElementType::FLOAT},
    };

    class GetResults
    {
      public:
        GetResults(std::size_t threshold, std::vector<TensorInfo> const& test_case)
        : id_map{},
          info_map{}
        {
          TensorFusor fusor {threshold};
          fusor.fuse_tensor_infos_and_ids(test_case, id_map, info_map);
        }

        TensorFusor::IDMap id_map;
        TensorFusor::InfoMap info_map;
    };

    class GetZeroThresholdReferenceResults
    {
      public:
        GetZeroThresholdReferenceResults(std::vector<TensorInfo> const& tensor_infos)
        : id_map(generate_id_map(tensor_infos)),
          info_map(generate_info_map(tensor_infos))
        { }

        TensorFusor::IDMap generate_id_map(std::vector<TensorInfo> const& tensor_infos)
        {
          TensorFusor::IDMap map {};
          for (auto const& tinfo : tensor_infos)
          {
            auto const id = tinfo.get_id();
            map[id] = id;
          }
          return map;
        }

        TensorFusor::InfoMap generate_info_map(std::vector<TensorInfo> const& tensor_infos)
        {
          TensorFusor::InfoMap map {};
          for (auto const& tinfo : tensor_infos)
          {
            auto const id = tinfo.get_id();
            map[id] = tinfo;
          }
          return map;
        }

        TensorFusor::IDMap id_map;
        TensorFusor::InfoMap info_map;
    };

    BOOST_AUTO_TEST_SUITE(tensor_fusor_unit)
      BOOST_DATA_TEST_CASE(tensor_fusor_with_zero_threshold, non_fusion_test_cases, test_case)
      {
        GetResults results {0UL, test_case};
        GetZeroThresholdReferenceResults reference {test_case};

        BOOST_TEST_REQUIRE(results.id_map == reference.id_map);
        BOOST_TEST_REQUIRE(results.info_map == reference.info_map);
      }

      BOOST_AUTO_TEST_CASE(tensor_fusor_with_threshold_2_floats)
      {
        GetResults results {2UL*4UL, fusion_test_case};
        GetZeroThresholdReferenceResults reference {fusion_test_case};

        BOOST_TEST_REQUIRE(results.id_map == reference.id_map);
        BOOST_TEST_REQUIRE(results.info_map == reference.info_map);
      }

      BOOST_AUTO_TEST_CASE(tensor_fusor_with_threshold_10_floats)
      {
        GetResults results {10UL*4UL, fusion_test_case};

        BOOST_TEST_REQUIRE(results.id_map.find(0UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(1UL)->second == 1UL);
        BOOST_TEST_REQUIRE(results.id_map.find(2UL)->second == 2UL);
        BOOST_TEST_REQUIRE(results.id_map.find(3UL)->second == 2UL);

        BOOST_TEST_REQUIRE(results.info_map.find(0UL)->second.get_nelems() == 94UL);
        BOOST_TEST_REQUIRE(results.info_map.find(1UL)->second.get_nelems() == 17UL);
        BOOST_TEST_REQUIRE(results.info_map.find(2UL)->second.get_nelems() == 83UL);
      }

      BOOST_AUTO_TEST_CASE(tensor_fusor_with_threshold_100_floats)
      {
        GetResults results {100UL*4UL, fusion_test_case};

        BOOST_TEST_REQUIRE(results.id_map.find(0UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(1UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(2UL)->second == 2UL);
        BOOST_TEST_REQUIRE(results.id_map.find(3UL)->second == 2UL);

        BOOST_TEST_REQUIRE(results.info_map.find(0UL)->second.get_nelems() == 111UL);
        BOOST_TEST_REQUIRE(results.info_map.find(2UL)->second.get_nelems() == 83UL);
      }

      BOOST_AUTO_TEST_CASE(tensor_fusor_with_threshold_112_floats)
      {
        GetResults results {112UL*4UL, fusion_test_case};

        BOOST_TEST_REQUIRE(results.id_map.find(0UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(1UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(2UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(3UL)->second == 3UL);

        BOOST_TEST_REQUIRE(results.info_map.find(0UL)->second.get_nelems() == 113UL);
        BOOST_TEST_REQUIRE(results.info_map.find(3UL)->second.get_nelems() == 81UL);
      }

      BOOST_AUTO_TEST_CASE(tensor_fusor_with_threshold_200_floats)
      {
        GetResults results {200UL*4UL, fusion_test_case};

        BOOST_TEST_REQUIRE(results.id_map.find(0UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(1UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(2UL)->second == 0UL);
        BOOST_TEST_REQUIRE(results.id_map.find(3UL)->second == 0UL);

        BOOST_TEST_REQUIRE(results.info_map.find(0UL)->second.get_nelems() == 194UL);
      }
    BOOST_AUTO_TEST_SUITE_END()
  }
}
