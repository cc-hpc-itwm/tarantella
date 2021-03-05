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
            new tarantella::TensorBroadcaster(group_all,
                                              tensor_infos,
                                              group_all.toGroupRank(root_rank)));
        }))
    .def("broadcast",
        [](tarantella::TensorBroadcaster &tb, std::vector<py::array>& tensor_list)
        {
          std::vector<void*> tensor_ptrs;
          for (auto& tens : tensor_list)
          {
            py::buffer_info info = tens.request(); 
            tensor_ptrs.emplace_back(info.ptr);
          }
          tb.exec_broadcast(tensor_ptrs);
        });

  py::class_<tarantella::TensorAllreducer>(m, "TensorAllreducer")
    .def(py::init(
        [](tarantella::GPI::Context& context,
           std::vector<tarantella::collectives::TensorInfo> tensor_infos)
        {
          tarantella::distribution::DataParallelGroupBuilder group_builder(context);
          tarantella::distribution::DataParallelSegmentIDBuilder segment_id_builder{};

          tarantella::collectives::Allreduce::ReductionOp reduction_op = tarantella::collectives::Allreduce::ReductionOp::SUM;

          return std::unique_ptr<tarantella::TensorAllreducer>(
            new tarantella::TensorAllreducer(context,
                                             segment_id_builder.get_segment_id(),
                                             group_builder.get_group(),
                                             reduction_op,
                                             tensor_infos));
        }),
        py::keep_alive<1, 2>())
    .def("allreduce",
        [](tarantella::TensorAllreducer &reducer,
           std::vector<py::array>& tensor_list)
        {
          std::vector<const void*> tensor_ptrs;

          std::vector<py::array> output_list(tensor_list.size());
          std::vector<void*> output_ptrs;

          for (std::size_t i = 0; i < tensor_list.size(); ++i)
          {
            py::buffer_info info = tensor_list[i].request();
            tensor_ptrs.emplace_back(info.ptr);

            auto src = py::array::ensure(tensor_list[i]);
            if (py::isinstance<py::array_t<float>>(src))
            {
              output_list[i] = py::array_t<float>(info.size);
            }
            else if (py::isinstance<py::array_t<double>>(src))
            {
              output_list[i] = py::array_t<double>(info.size);
            }            
            else if (py::isinstance<py::array_t<std::int32_t>>(src))
            {
              output_list[i] = py::array_t<std::int32_t>(info.size);
            }
            else if (py::isinstance<py::array_t<std::int16_t>>(src))
            {
              output_list[i] = py::array_t<std::int16_t>(info.size);
            }
            else
            {
              throw std::runtime_error("[Pybind11][TensorAllreducer] Unknown buffer type");
            }

            py::buffer_info output_buffer = output_list[i].request();
            output_ptrs.emplace_back(output_buffer.ptr);
          }
          
          reducer.exec_allreduce(tensor_ptrs, output_ptrs);
          return output_list;
        });

  py::class_<gaspi::collectives::blocking::Barrier>(m, "Barrier")
    .def(py::init(
        []()
        {
          gaspi::group::Group group_all;
          return std::make_unique<gaspi::collectives::blocking::Barrier>(group_all);
        }))
    .def("blocking_barrier_all_ranks", &gaspi::collectives::blocking::Barrier::execute);

  py::class_<tarantella::PipelineCommunicator>(m, "PipelineCommunicator")
    .def(py::init(
        [](std::unordered_map<tarantella::PipelineCommunicator::ConnectionID,
                              std::pair<std::pair<gaspi::group::GlobalRank, gaspi::group::GlobalRank>, std::size_t>> edges,
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
