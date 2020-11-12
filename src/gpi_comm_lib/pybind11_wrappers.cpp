#include "collectives/BufferElementType.hpp"
#include "collectives/TensorInfo.hpp"
#include "distribution/GroupBuilder.hpp"
#include "distribution/SegmentIDBuilder.hpp"
#include "gpi/Context.hpp"
#include "PipelineCommunicator.hpp"
#include "SynchCommunicator.hpp"
#include "TensorBroadcaster.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(GPICommLib, m)
{
  m.doc() = "GPI communication library for Deep Learning";

  py::class_<tarantella::GPI::Context>(m, "GPIContext")
      .def(py::init<>())
      .def_property_readonly("rank", &tarantella::GPI::Context::get_rank)
      .def_property_readonly("size", &tarantella::GPI::Context::get_comm_size);

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
        [](tarantella::GPI::Context& context,
           std::vector<tarantella::collectives::TensorInfo> tensor_infos,
           std::size_t fusion_threshold_bytes)
        {
          tarantella::distribution::DataParallelGroupBuilder group_builder(context);
          tarantella::distribution::DataParallelSegmentIDBuilder segment_id_builder{};

          return std::unique_ptr<tarantella::SynchCommunicator>(
            new tarantella::SynchCommunicator(context,
                                              segment_id_builder.get_segment_id(),
                                              group_builder.get_group(),
                                              tensor_infos,
                                              fusion_threshold_bytes));
        }),
        // ensure the `context` object is not garbage-collected as long as the SynchCommunicator is alive
        py::keep_alive<1, 2>())
    .def("get_raw_ptr", [](tarantella::SynchCommunicator& d) 
        {
          return reinterpret_cast<uint64_t>(&d);
        },
        py::return_value_policy::reference_internal);

  py::class_<tarantella::TensorBroadcaster>(m, "TensorBroadcaster")
    .def(py::init(
        [](tarantella::GPI::Context& context,
           std::vector<tarantella::collectives::TensorInfo> tensor_infos,
           tarantella::GPI::Rank root_rank)
        {
          tarantella::distribution::DataParallelGroupBuilder group_builder(context);
          tarantella::distribution::DataParallelSegmentIDBuilder segment_id_builder{};

          return std::unique_ptr<tarantella::TensorBroadcaster>(
            new tarantella::TensorBroadcaster(context,
                                              segment_id_builder.get_segment_id(),
                                              group_builder.get_group(),
                                              tensor_infos,
                                              root_rank));
        }),
        py::keep_alive<1, 2>())
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

  py::class_<tarantella::collectives::Barrier::GPIBarrierAllRanks>(m, "Barrier")
    .def(py::init(
        [](tarantella::GPI::Context&)
        {
          return std::unique_ptr<tarantella::collectives::Barrier::GPIBarrierAllRanks>(
            new tarantella::collectives::Barrier::GPIBarrierAllRanks());
        }),
        py::keep_alive<1, 2>())
    .def("blocking_barrier_all_ranks",
        [](tarantella::collectives::Barrier::GPIBarrierAllRanks &barrier)
        {
          barrier.blocking_barrier();
        });

  py::class_<tarantella::PipelineCommunicator>(m, "PipelineCommunicator")
    .def(py::init(
        [](tarantella::GPI::Context& context, 
           std::unordered_map<tarantella::PipelineCommunicator::ConnectionID,
                              std::pair<std::pair<tarantella::GPI::Rank, tarantella::GPI::Rank>, std::size_t>> edges,
           std::size_t num_micro_batches)
        {
          auto const rank = context.get_rank();
          std::unordered_map<tarantella::PipelineCommunicator::ConnectionID,
                             tarantella::ConnectionInfo> conn_infos;
          tarantella::distribution::PipelineSegmentIDBuilder segment_id_builder;

          // build connection info (segment_id, other rank, buffer_size)
          // for each edge connected to the current rank
          for (auto const& [conn_id, edge_and_size] : edges)
          {
            auto const ranks = edge_and_size.first;
            if (ranks.first != rank && ranks.second != rank) continue;
            
            auto const other_rank = (ranks.first == rank) ? ranks.second : ranks.first;
            auto const buffer_size = edge_and_size.second;
            auto const segment_id = segment_id_builder.get_segment_id(conn_id);
            tarantella::ConnectionInfo const conn_info(segment_id, other_rank, buffer_size);
            conn_infos.emplace(conn_id, conn_info);
          }

          return std::make_unique<tarantella::PipelineCommunicator>(context, conn_infos, num_micro_batches);
        }),
        // ensure the `context` object is not garbage-collected as long as the PipelineCommunicator is alive
        py::keep_alive<1, 2>())
    .def("get_raw_ptr", [](tarantella::PipelineCommunicator& comm) 
        {
          return reinterpret_cast<uint64_t>(&comm);
        },
        py::return_value_policy::reference_internal);
}
