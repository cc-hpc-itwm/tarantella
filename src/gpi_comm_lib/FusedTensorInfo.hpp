#pragma once

#include "BufferElementType.hpp"
#include "TensorInfo.hpp"
#include "Types.hpp"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace tarantella
{
  namespace collectives
  {
    class FusedTensorInfo
    {
      public:
        FusedTensorInfo();
        FusedTensorInfo(TensorInfo const&);
        FusedTensorInfo& operator=(TensorInfo const&);
        bool operator==(FusedTensorInfo const&) const;

        FusedID get_id() const;
        std::size_t get_nelems() const;
        BufferElementType get_elem_type() const;
        std::size_t get_size_bytes() const;

        std::size_t get_num_tensors() const;
        std::vector<GradID> get_tensor_ids() const;

        std::size_t get_local_offset_bytes(GradID const&) const;
        std::size_t get_local_size_bytes(GradID const&) const;
      
        void add_tensor_info(TensorInfo const&);
        TensorInfo to_tensor_info() const;

      private:
        FusedID id;
        std::size_t nelems;
        BufferElementType elem_type;
        std::size_t elem_size;
        std::size_t size_bytes;
        std::size_t num_tensors;

        std::vector<GradID> tensor_ids;
        std::unordered_map<GradID, std::size_t> local_offset_bytes;
        std::unordered_map<GradID, std::size_t> local_size_bytes;

        void initialise_from_tensor_info(TensorInfo const&);
    };

    class TensorFusor
    {
      public:
        using IDMap = std::unordered_map<GradID, FusedID>;
        using InfoMap = std::unordered_map<FusedID, collectives::FusedTensorInfo>;

        TensorFusor();
        TensorFusor(std::size_t threshold);

        void fuse_tensor_infos_and_ids(std::vector<TensorInfo> const&,
                                       IDMap&,
                                       InfoMap&);

      private:
        std::size_t threshold_bytes;
    };
  }
}
