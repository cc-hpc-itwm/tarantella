#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "PipelineCommunicator.hpp"

using namespace tensorflow;

REGISTER_OP("SendOp")
    .Attr("tnt_pipeline_comm: int")
    .Input("input_tensor: float")
    .Input("connection_id: int32")
    .Input("micro_batch_id: int32")
    .Output("out_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) 
    {
      c->set_output(0, c->input(0)); 
      return Status::OK();
    });
REGISTER_OP("RecvOp")
    .Attr("tnt_pipeline_comm: int")
    .Input("input_tensor: float")
    .Input("connection_id: int32")
    .Input("micro_batch_id: int32")
    .Output("out_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) 
    {
      c->set_output(0, c->input(0)); 
      return Status::OK();
    });

class SendOp : public OpKernel
{
  public:
    explicit SendOp(OpKernelConstruction* context)
    : OpKernel(context)
    {
      tensorflow::int64 context_ptr;
      OP_REQUIRES_OK(context, context->GetAttr("tnt_pipeline_comm", &context_ptr));
      pipeline_communicator = reinterpret_cast<tarantella::PipelineCommunicator*>(context_ptr);
    }

    void Compute(OpKernelContext* context) override
    {
      const Tensor& input_tensor = context->input(0);
      const Tensor& conn_id_tensor = context->input(1);
      const Tensor& micro_batch_id_tensor = context->input(2);

      auto send_buf = reinterpret_cast<void*>(const_cast<float*>(input_tensor.flat<float>().data()));
      auto const conn_id = static_cast<tarantella::PipelineCommunicator::ConnectionID>(
                                                  conn_id_tensor.flat<int>().data()[0]);
      auto const micro_batch_id = static_cast<tarantella::PipelineCommunicator::MicrobatchID>(
                                                  micro_batch_id_tensor.flat<int>().data()[0]);

      // allocate (fake) output
      auto const output_index = 0;
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(output_index, input_tensor.shape(), &output_tensor));

      pipeline_communicator->non_blocking_send(send_buf, conn_id, micro_batch_id);
    }

  private:
    tarantella::PipelineCommunicator *pipeline_communicator;
};

class RecvOp : public OpKernel
{
  public:
    explicit RecvOp(OpKernelConstruction* context)
    : OpKernel(context)
    {
      tensorflow::int64 context_ptr;
      OP_REQUIRES_OK(context, context->GetAttr("tnt_pipeline_comm", &context_ptr));
      pipeline_communicator = reinterpret_cast<tarantella::PipelineCommunicator *>(context_ptr);
    }

    void Compute(OpKernelContext* context) override
    {
      const Tensor& input_tensor = context->input(0);
      const Tensor& conn_id_tensor = context->input(1);
      const Tensor& micro_batch_id_tensor = context->input(2);

      auto const conn_id = static_cast<tarantella::PipelineCommunicator::ConnectionID>(
                                                  conn_id_tensor.flat<int>().data()[0]);
      auto const micro_batch_id = static_cast<tarantella::PipelineCommunicator::MicrobatchID>(
                                                  micro_batch_id_tensor.flat<int>().data()[0]);

      // allocate output
      auto const output_index = 0;
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(output_index, input_tensor.shape(), &output_tensor));

      auto* recv_buf = output_tensor->flat<float>().data();
      pipeline_communicator->blocking_recv(recv_buf, conn_id, micro_batch_id);
  }

  private:
    tarantella::PipelineCommunicator *pipeline_communicator;
};

REGISTER_KERNEL_BUILDER(Name("SendOp").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("RecvOp").Device(DEVICE_CPU), RecvOp);