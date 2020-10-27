#pragma once

#include "collectives/barrier/GPIBarrier.hpp"
#include "collectives/TensorInfo.hpp"
#include "gpi/Context.hpp"
#include "gpi/Group.hpp"
#include "gpi/SegmentBuffer.hpp"
#include "broadcast.h"

#include <memory>
#include <vector>

namespace tarantella
{

  class TensorBroadcaster
  {
    public:
      TensorBroadcaster(GPI::Context&, GPI::SegmentID, GPI::Group const&,
                        std::vector<collectives::TensorInfo> const&, GPI::Rank root_rank);
      void exec_broadcast(std::vector<void*> const&);

    private:
      GPI::Context& context;
      GPI::Group const group;
      collectives::queues queue_handler; // FIXME: use GPI::ResourcesManager
      GPI::Rank root;
      collectives::Barrier::GPIBarrier barrier;

      std::vector<GPI::SegmentBuffer> buffers;
      std::unique_ptr<collectives::broadcast> bcast_op;
  };
}
