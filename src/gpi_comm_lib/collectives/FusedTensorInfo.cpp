#include "FusedTensorInfo.hpp"

namespace tarantella
{
  namespace collectives
  {
    void FusedTensorInfo::initialise_from_tensor_info(TensorInfo const& tensor_info)
    {
      local_offset_bytes.clear();
      local_size_bytes.clear();

      id = tensor_info.get_id();
      nelems = tensor_info.get_nelems();
      elem_type = tensor_info.get_elem_type();
      elem_size = getDataTypeSize(elem_type);
      size_bytes = nelems * elem_size;
      num_tensors = 1UL;
      tensor_ids.push_back(id);
      local_offset_bytes[id] = 0UL;
      local_size_bytes[id] = size_bytes;
    }

    FusedTensorInfo::FusedTensorInfo()
    : id(),
      nelems(),
      elem_type(),
      elem_size(),
      size_bytes(),
      num_tensors(),
      tensor_ids(),
      local_offset_bytes(),
      local_size_bytes()
    { }

    FusedTensorInfo::FusedTensorInfo(TensorInfo const& tensor_info)
    : FusedTensorInfo()
    {
      initialise_from_tensor_info(tensor_info);
    }

    FusedTensorInfo& FusedTensorInfo::operator=(TensorInfo const& tensor_info)
    {
      initialise_from_tensor_info(tensor_info);
      return *this;
    }

    bool FusedTensorInfo::operator==(FusedTensorInfo const& other) const
    {
      return ( this->id == other.id &&
               this->nelems == other.nelems &&
               this->elem_type == other.elem_type &&
               this->num_tensors == other.num_tensors &&
               this->local_offset_bytes == other.local_offset_bytes &&
               this->local_size_bytes == other.local_size_bytes );

    }

    FusedID FusedTensorInfo::get_id() const
    {
      return id;
    }

    std::size_t FusedTensorInfo::get_nelems() const
    {
      return nelems;
    }

    BufferElementType FusedTensorInfo::get_elem_type() const
    {
      return elem_type;
    }

    std::size_t FusedTensorInfo::get_size_bytes() const
    {
      return size_bytes;
    }

    std::size_t FusedTensorInfo::get_num_tensors() const
    {
      return num_tensors;
    }

    std::vector<GradID> FusedTensorInfo::get_tensor_ids() const
    {
      return tensor_ids;
    }

    std::size_t FusedTensorInfo::get_local_offset_bytes(GradID const& grad_id) const
    {
      auto const it = local_offset_bytes.find(grad_id);
      if (it == local_offset_bytes.end())
      {
        throw std::logic_error("FusedTensorInfo::get_local_offset_bytes: FusedTensorInfo does not contain GradID");
      }
      return it->second;
    }

    std::size_t FusedTensorInfo::get_local_size_bytes(GradID const& grad_id) const
    {
      auto const it = local_size_bytes.find(grad_id);
      if (it == local_size_bytes.end())
      {
        throw std::logic_error("FusedTensorInfo::get_local_size_bytes: FusedTensorInfo does not contain GradID");
      }
      return it->second;
    }

    void FusedTensorInfo::add_tensor_info(TensorInfo const& tensor_info)
    {
      if (tensor_info.get_elem_type() != get_elem_type())
      {
        throw std::logic_error("FusedTensorInfo::add_tensor_info: Tensors need to have same data type");
      }

      auto const grad_id = tensor_info.get_id();
      auto const grad_nelems = tensor_info.get_nelems();
      auto const grad_size_bytes = grad_nelems * elem_size;
      auto const current_offset = size_bytes;

      nelems += grad_nelems;
      size_bytes += grad_size_bytes;
      num_tensors += 1UL;

      tensor_ids.push_back(grad_id);
      local_offset_bytes[grad_id] = current_offset;
      local_size_bytes[grad_id] = grad_size_bytes;
    }

    TensorInfo FusedTensorInfo::to_tensor_info() const
    {
      return {get_id(), get_nelems(), get_elem_type()};
    }

    TensorFusor::TensorFusor()
    : threshold_bytes(0UL)
    { }

    TensorFusor::TensorFusor(std::size_t threshold)
    : threshold_bytes(threshold)
    { }

    void TensorFusor::fuse_tensor_infos_and_ids(std::vector<TensorInfo> const& tensor_infos,
                                                IDMap& fused_ids,
                                                InfoMap& fused_tensor_infos)
    {
      if (tensor_infos.size() == 1)
      {
        auto const tensor_info = tensor_infos.front();
        auto const id = tensor_info.get_id();
        fused_ids[id] = id;
        fused_tensor_infos[id] = tensor_info;
      }

      collectives::FusedTensorInfo fused_info(tensor_infos.front());
      auto tensor_id = tensor_infos.front().get_id();
      FusedID fused_id(tensor_id);
      fused_ids[tensor_id] = fused_id;

      for (auto idx = 1UL; idx < tensor_infos.size(); ++idx)
      {
        tensor_id = tensor_infos[idx].get_id();

        if (fused_info.get_size_bytes() < threshold_bytes)
        {
          fused_info.add_tensor_info(tensor_infos[idx]);
        }
        else
        {
          fused_tensor_infos[fused_id] = fused_info;
          fused_id = tensor_id;
          fused_info = tensor_infos[idx];
        }

        fused_ids[tensor_id] = fused_id;

        // Always add the last fused_tensor to the vector.
        // Note, that it might still be smaller than `threshold_bytes`.
        if (idx == tensor_infos.size() - 1)
        {
          fused_tensor_infos[fused_id] = fused_info;
        }
      }
    }
  }
}
