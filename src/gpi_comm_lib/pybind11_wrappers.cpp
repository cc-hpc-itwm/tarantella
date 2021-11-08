#include "BufferElementType.hpp"
#include "TensorInfo.hpp"
#include "PipelineCommunicator.hpp"
#include "SynchCommunicator.hpp"
#include "TensorBroadcaster.hpp"
#include "TensorAllreducer.hpp"

#include <GaspiCxx/Runtime.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(GPICommLib, m)
{
  m.doc() = "GPI communication library for Deep Learning";

  m.def("initGaspiCxx", []()
                {
                  gaspi::initGaspiCxx();
                });
  m.def("get_rank", []()
                {
                  return gaspi::getRuntime().global_rank();
                });
  m.def("get_size", []()
                {
                  return gaspi::getRuntime().size();
                });

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
          return std::unique_ptr<tarantella::collectives::TensorInfo>(
                             new tarantella::collectives::TensorInfo(tensid, nelems, elemtype));
        }));

  py::class_<tarantella::SynchCommunicator>(m, "SynchDistCommunicator")
    .def(py::init(
        [](std::vector<tarantella::collectives::TensorInfo> tensor_infos,
           std::size_t fusion_threshold_bytes)
        {
          gaspi::group::Group group_all;
          return std::unique_ptr<tarantella::SynchCommunicator>(
            new tarantella::SynchCommunicator(group_all,
                                              tensor_infos,
                                              fusion_threshold_bytes));
        }))
    .def("get_raw_ptr", [](tarantella::SynchCommunicator& d) 
        {
          return reinterpret_cast<uint64_t>(&d);
        },
        py::return_value_policy::reference_internal);

  py::class_<tarantella::TensorBroadcaster>(m, "TensorBroadcaster")
    .def(py::init(
        [](std::vector<tarantella::collectives::TensorInfo> tensor_infos,
           gaspi::group::GlobalRank root_rank)
        {
          gaspi::group::Group group_all;
          return std::unique_ptr<tarantella::TensorBroadcaster>(
            new tarantella::TensorBroadcaster(tensor_infos,
                                              group_all,
                                              group_all.toGroupRank(root_rank)));
        }))
    .def("broadcast",
        [](tarantella::TensorBroadcaster& tensor_broadcaster, std::vector<py::array>& input_list)
        {
          // allocate memory for outputs
          std::vector<py::array> output_list;
          for (auto const& output_size : tensor_broadcaster.get_sizes())
          {
            output_list.push_back(py::array_t<float>(output_size));
          }

          // extract output pointers
          std::vector<void*> output_ptrs;
          for (auto const& output : output_list)
          {
            output_ptrs.push_back(output.request().ptr);
          }

          // extract pointers for inputs if `input_list` is not empty
          // otherwise create list of as many `nullptr`s as outputs
          std::vector<void const*> input_ptrs(output_ptrs.size());
          if (input_list.size() > 0)
          {
            for (auto i = 0UL; i < input_ptrs.size(); ++i)
            {
              input_ptrs[i] = input_list[i].request().ptr;
            }
          }

          tensor_broadcaster.exec_broadcast(input_ptrs, output_ptrs);
          return output_list;
        });

  py::class_<tarantella::TensorAllreducer>(m, "TensorAllreducer")
    .def(py::init(
        [](std::vector<tarantella::collectives::TensorInfo> tensor_infos)
        {
          gaspi::collectives::ReductionOp reduction_op = gaspi::collectives::ReductionOp::SUM;
          gaspi::group::Group group_all;

          return std::unique_ptr<tarantella::TensorAllreducer>(
            new tarantella::TensorAllreducer(tensor_infos,
                                             group_all,
                                             reduction_op));
        }))
    .def("allreduce",
        [](tarantella::TensorAllreducer& tensor_allreducer, std::vector<py::array>& input_list)
        {
          // allocate memory for outputs
          std::vector<py::array> output_list;
          for (auto const& input : input_list)
          {
            auto const info = input.request();
            if (py::isinstance<py::array_t<float>>(py::array::ensure(input)))
            {
              output_list.push_back(py::array_t<float>(info.size));
            }
            else if (py::isinstance<py::array_t<double>>(py::array::ensure(input)))
            {
              output_list.push_back(py::array_t<double>(info.size));
            }
            else if (py::isinstance<py::array_t<int64_t>>(py::array::ensure(input)))
            {
              output_list.push_back(py::array_t<int64_t>(info.size));
            }
            else if (py::isinstance<py::array_t<int16_t>>(py::array::ensure(input)))
            {
              output_list.push_back(py::array_t<int16_t>(info.size));
            }
            else if (py::isinstance<py::array_t<int32_t>>(py::array::ensure(input)))
            {
              output_list.push_back(py::array_t<int32_t>(info.size));
            }
          }

          // extract pointers for inputs and outputs
          std::vector<void const*> input_ptrs;
          for (auto const& input : input_list)
          {
            input_ptrs.push_back(input.request().ptr);
          }
          std::vector<void*> output_ptrs;
          for (auto const& output : output_list)
          {
            output_ptrs.push_back(output.request().ptr);
          }

          tensor_allreducer.exec_allreduce(input_ptrs, output_ptrs);
          return output_list;
        });

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
