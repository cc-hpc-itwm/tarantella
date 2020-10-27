#pragma once

#include "collectives/TensorInfo.hpp"

#include <cstddef>
#include <vector>

namespace tarantella
{
  namespace distribution
  {
    std::size_t get_segment_size(std::vector<collectives::TensorInfo> const& DNN, double overhead_factor);
  }
}

