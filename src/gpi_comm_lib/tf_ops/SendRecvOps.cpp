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
REGISTER_OP("SendWithAckOp")
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
REGISTER_OP("RecvWithAckOp")
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

template<typename P2Pimplementation>
class P2POp : public OpKernel
{
  public:
    using ConnectionID = tarantella::PipelineCommunicator::ConnectionID;
    using MicrobatchID = tarantella::PipelineCommunicator::MicrobatchID;

    explicit P2POp(OpKernelConstruction* context)
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

      auto const conn_id = static_cast<ConnectionID>(conn_id_tensor.flat<int>().data()[0]);
      auto const micro_batch_id = static_cast<MicrobatchID>(
                                              micro_batch_id_tensor.flat<int>().data()[0]);

      // allocate output
      auto const output_index = 0;
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(output_index, input_tensor.shape(), &output_tensor));

      P2Pimplementation& impl = static_cast<P2Pimplementation&>(*this);
      impl.execute_op(input_tensor, *output_tensor, conn_id, micro_batch_id);
    }

  protected:
    tarantella::PipelineCommunicator *pipeline_communicator;
};

class SendOp : public P2POp<SendOp>
{
  public:
    using P2POp<SendOp>::P2POp;

    void execute_op(Tensor const& input_tensor, Tensor&,
                    ConnectionID conn_id, MicrobatchID micro_batch_id)
    {
      auto send_buf = reinterpret_cast<void*>(const_cast<float*>(input_tensor.flat<float>().data()));
      pipeline_communicator->non_blocking_send(send_buf, conn_id, micro_batch_id);
    }
};

class RecvOp : public P2POp<RecvOp>
{
  public:
    using P2POp<RecvOp>::P2POp;

    void execute_op(Tensor const&, Tensor& output_tensor,
                    ConnectionID conn_id, MicrobatchID micro_batch_id)
    {
      auto* recv_buf = output_tensor.flat<float>().data();
      pipeline_communicator->blocking_recv(recv_buf, conn_id, micro_batch_id);
    }
};

class SendWithAckOp : public P2POp<SendWithAckOp>
{
  public:
    using P2POp<SendWithAckOp>::P2POp;

    void execute_op(Tensor const& input_tensor, Tensor&,
                    ConnectionID conn_id, MicrobatchID micro_batch_id)
    {
      auto send_buf = reinterpret_cast<void*>(const_cast<float*>(input_tensor.flat<float>().data()));
      pipeline_communicator->send_with_acknowledgement(send_buf, conn_id, micro_batch_id);
    }
};

class RecvWithAckOp : public P2POp<RecvWithAckOp>
{
  public:
    using P2POp<RecvWithAckOp>::P2POp;

    void execute_op(Tensor const&, Tensor& output_tensor,
                    ConnectionID conn_id, MicrobatchID micro_batch_id)
    {
      auto* recv_buf = output_tensor.flat<float>().data();
      pipeline_communicator->recv_with_acknowledgement(recv_buf, conn_id, micro_batch_id);
    }
};

REGISTER_KERNEL_BUILDER(Name("SendOp").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("RecvOp").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(Name("SendWithAckOp").Device(DEVICE_CPU), SendWithAckOp);
REGISTER_KERNEL_BUILDER(Name("RecvWithAckOp").Device(DEVICE_CPU), RecvWithAckOp);