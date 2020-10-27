#include "GPIBarrier.hpp"
#include "gpi/gaspiCheckReturn.hpp"

#include <stdexcept>

namespace tarantella
{
  namespace collectives
  {
    namespace Barrier
    {
      GPIBarrier::GPIBarrier(GPI::Group const &group)
      {
        gaspi_rank_t comm_size;
        GPI::gaspiCheckReturn(gaspi_proc_num(&comm_size),
                              "GPIBarrier::GPIBarrier : get number of ranks");
        if (group.get_size() != comm_size)
        {
          throw std::invalid_argument("GPIBarrier::GPIBarrier : can only be used with all ranks in \
                                      the default GPI communicator");
        }
      }

      // TODO: implement for any GPI::Group
      void GPIBarrier::blocking_barrier()
      {
        GPI::gaspiCheckReturn(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK),
                              "GPIBarrier::GPIBarrier : barrier failed");
      }

      void GPIBarrierAllRanks::blocking_barrier()
      {
        GPI::gaspiCheckReturn(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK),
                              "GPIBarrierAllRanks::GPIBarrierAllRanks : barrier failed");
      }
    } 
  }   
} 
