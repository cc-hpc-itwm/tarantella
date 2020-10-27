#include "utilities.hpp"

#include <numeric>
#include <stdexcept>

namespace tarantella
{
  namespace distribution
  {
    std::size_t get_segment_size(std::vector<collectives::TensorInfo> const& DNN, double overhead_factor)
    {
      if(DNN.size() == 0)
      {
        throw std::logic_error("tarantella::get_segment_size: Empty DNN to SynchCommunicator provided");
      }

      auto add_tensor_size_in_bytes = [](auto sum, auto tensor_info){
        return sum + (tensor_info.get_nelems() * getDataTypeSize(tensor_info.get_elem_type())); };
      auto const partition_size = std::accumulate(DNN.begin(), DNN.end(), 0UL, add_tensor_size_in_bytes);
      return overhead_factor * partition_size;
    }
  }
}
