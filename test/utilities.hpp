#include "TensorInfo.hpp"

#include <GaspiCxx/group/Group.hpp>

#include <numeric>
#include <vector>

namespace tarantella
{
  std::vector<gaspi::group::GlobalRank> gen_group_ranks(size_t nranks_in_group)
  {
    std::vector<gaspi::group::GlobalRank> group_ranks(nranks_in_group);
    std::iota(group_ranks.begin(), group_ranks.end(), 0);
    return group_ranks;
  }
}

namespace std
{
  std::ostream& operator<< (std::ostream& os, const std::vector<tarantella::collectives::TensorInfo>& tlist)
  {
    for (auto& tinfo : tlist)
    {
      os << "TensorID=" << tinfo.get_id() << " nelems=" << tinfo.get_nelems()
         << " dtype_size=" << getDataTypeSize(tinfo.get_elem_type())<< std::endl;
    }
    return os;
  }
}