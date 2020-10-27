#include "Context.hpp"

#include "gpi/gaspiCheckReturn.hpp"
#include "gpi/Group.hpp"
#include "gpi/ResourceManager.hpp"

#include <algorithm>
#include <numeric>

namespace tarantella
{
  namespace GPI
  {
    Context::Context()
    : rank(0), comm_size(0)
    {
      gaspiCheckReturn(gaspi_proc_init(GASPI_BLOCK),
                       "GPI library initialization");
      gaspiCheckReturn(gaspi_proc_rank(&rank),
                       "get rank");
      gaspi_rank_t size; // gaspi_proc_num expects gaspi_rank_t
      gaspiCheckReturn(gaspi_proc_num(&size),
                       "get number of processes");
      comm_size = size;
    }

    Context::~Context()
    {
      gaspiCheckReturn(gaspi_barrier(GASPI_GROUP_ALL, timeout_millis),
                      "gaspi_barrier");
      gaspiCheckReturn(gaspi_proc_term(GASPI_BLOCK),
                       "GPI library finalize");
    }

    Rank Context::get_rank() const
    {
      return rank;
    }

    std::size_t Context::get_comm_size() const
    {
      return comm_size;
    }

    tarantella::GPI::ResourceManager& Context::get_resource_manager()
    {
      return tarantella::GPI::ResourceManager::get_instance(*this);
    }

    void Context::allocate_segment(SegmentID id, Group const& group, std::size_t total_size)
    {
      if (total_size == 0)
      {
        throw std::runtime_error("Context::allocate_segment : Cannot allocate segment of size zero");
      }

      if (!group.contains_rank(get_rank()))
      {
        throw std::runtime_error("Context::allocate_segment : Group does not contain rank");
      }

      gaspiCheckReturn(gaspi_segment_alloc(id, total_size, GASPI_MEM_UNINITIALIZED),
                       "Context::allocate_segment : segment could not be allocated");
      for (auto other_rank : group.get_ranks())
      {
        if (other_rank != get_rank())
        {
          gaspiCheckReturn(gaspi_segment_register(id, other_rank, GASPI_BLOCK),
                           "Context::allocate_segment : segment could not be registered");
        }
      }
    }

    void Context::deallocate_segment(SegmentID id, Group const& group)
    {
      if (!group.contains_rank(get_rank()))
      {
        throw std::runtime_error("Context::deallocate_segment : Group does not contain rank");
      }
      gaspiCheckReturn(gaspi_segment_delete(id),
                       "Context::deallocate_segment : segment could not be deleted");
    }

    void* Context::get_segment_pointer(SegmentID id) const
    {
      void* p;
      gaspiCheckReturn(gaspi_segment_ptr(id, &p), "get pointer within segment");
      return p;
    }
  }
}

