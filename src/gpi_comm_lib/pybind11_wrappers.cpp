#include "BufferElementType.hpp"
#include "TensorInfo.hpp"
#include "PipelineCommunicator.hpp"
#include "SynchCommunicator.hpp"

#include <GaspiCxx/Runtime.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(GPICommLib, m)
{
  m.doc() = "Interface to communication managers for data and model parallelism";

  py::class_<tarantella::collectives::TensorInfo>(m, "TensorInfo")
    .def(py::init(
        [](std::size_t tensid, std::size_t nelems, py::dtype tensdtype)
        {
          tarantella::collectives::BufferElementType elemtype;
          if (tensdtype.is(py::dtype::of<float>()))
          {
            elemtype = tarantella::collectives::BufferElementType::FLOAT;
          }
          else if (tensdtype.is(py::dtype::of<double>()))
          {
            elemtype = tarantella::collectives::BufferElementType::DOUBLE;
          }
          else if (tensdtype.is(py::dtype::of<int64_t>()))
          {
            elemtype = tarantella::collectives::BufferElementType::INT64;
          }
          else if (tensdtype.is(py::dtype::of<int32_t>()))
          {
            elemtype = tarantella::collectives::BufferElementType::INT32;
          }
          else if (tensdtype.is(py::dtype::of<std::int16_t>()))
          {
            elemtype = tarantella::collectives::BufferElementType::INT16;
          }
          else
          {
            throw std::runtime_error("[Pybind11][TensorInfo] Unknown buffer type");
          }
          return std::make_unique<tarantella::collectives::TensorInfo>(tensid, nelems, elemtype);
        }));

  py::class_<tarantella::SynchCommunicator>(m, "SynchDistCommunicator")
    .def(py::init(
        [](std::vector<tarantella::collectives::TensorInfo> tensor_infos,
           std::size_t fusion_threshold_bytes)
        {
          gaspi::group::Group group_all;
          return std::make_unique<tarantella::SynchCommunicator>(group_all,
                                                                 tensor_infos,
                                                                 fusion_threshold_bytes);
        }))
    .def("get_raw_ptr", [](tarantella::SynchCommunicator& d) 
        {
          return reinterpret_cast<uint64_t>(&d);
        },
        py::return_value_policy::reference_internal);


  py::class_<tarantella::PipelineCommunicator>(m, "PipelineCommunicator")
    .def(py::init(
        [](std::unordered_map<tarantella::PipelineCommunicator::ConnectionID,
                              std::pair<gaspi::group::GlobalRank, std::size_t>> edges,
           std::size_t num_micro_batches)
        {
          return std::make_unique<tarantella::PipelineCommunicator>(edges, num_micro_batches);
        }))
    .def("get_raw_ptr", [](tarantella::PipelineCommunicator& comm) 
        {
          return reinterpret_cast<uint64_t>(&comm);
        },
        py::return_value_policy::reference_internal);
}
