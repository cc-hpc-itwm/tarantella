#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "SynchCommunicator.hpp"

using namespace tensorflow;

REGISTER_OP("StartAllreduceOp")
    .Attr("tnt_synchcomm: int")
    .Attr("tensor_id: int")
    .Input("input_tensor: float")
    .Output("out_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
REGISTER_OP("FinishAllreduceOp")
    .Attr("tnt_synchcomm: int")
    .Attr("tensor_id: int")
    .Attr("Tout: type")
    .Input("input_tensor: float")
    .Output("out_tensor: Tout")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
REGISTER_OP("BarrierOp")
    .Attr("T: list(type)")
    .Attr("Tout: list(type)")
    .Input("in: T")
    .Output("out: Tout")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {       
      for (auto i = 0; i < c->num_outputs(); ++i)
      {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

template <typename T>
class CommunicateTensorOp : public OpKernel
{
  public:
    explicit CommunicateTensorOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
      tensorflow::int64 context_ptr;
      OP_REQUIRES_OK(context,
                    context->GetAttr("tnt_synchcomm", &context_ptr));
      synch_communicator = reinterpret_cast<tarantella::SynchCommunicator *>(context_ptr);
      OP_REQUIRES_OK(context,
                     context->GetAttr("tensor_id", &tensor_id));
    }

    void Compute(OpKernelContext* context) override
    {
      static_cast<T&>(*this).compute_impl(context);
    }

  protected:
    tensorflow::int64 tensor_id;
    tarantella::SynchCommunicator *synch_communicator;
};

class StartAllreduceOp : public CommunicateTensorOp<StartAllreduceOp>
{
  public:
    explicit StartAllreduceOp(OpKernelConstruction* context)
    : CommunicateTensorOp<StartAllreduceOp>(context)
    { }

    void compute_impl(OpKernelContext* context)
    {
      auto input_index = 0;
      auto output_index = 0;
      const Tensor &input_tensor = context->input(input_index);
      auto* input_flat = input_tensor.flat<float>().data();

      synch_communicator->start_allreduce_impl(tensor_id, input_flat);
      context->set_output(output_index, input_tensor);
    }
    
};

class FinishAllreduceOp : public CommunicateTensorOp<FinishAllreduceOp>
{
  public:
    explicit FinishAllreduceOp(OpKernelConstruction* context)
    : CommunicateTensorOp<FinishAllreduceOp>(context)
    { }

    void compute_impl(OpKernelContext *context)
    {
      auto input_index = 0;
      auto output_index = 0;
      const Tensor &input_tensor = context->input(input_index);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(output_index, input_tensor.shape(),
                                                       &output_tensor));
      auto* output_flat = output_tensor->flat<float>().data();
      synch_communicator->finish_allreduce_impl(tensor_id, output_flat);
    }
};

class BarrierOp : public OpKernel
{
  public:
    explicit BarrierOp(OpKernelConstruction* context)
    : OpKernel(context)
    {}

    void Compute(OpKernelContext* context) override
    {
      for (auto i = 0; i < context->num_outputs(); ++i)
      {
        context->set_output(i, context->input(i));
      }
    }
};

REGISTER_KERNEL_BUILDER(Name("StartAllreduceOp").Device(DEVICE_CPU), StartAllreduceOp);
REGISTER_KERNEL_BUILDER(Name("FinishAllreduceOp").Device(DEVICE_CPU), FinishAllreduceOp);
REGISTER_KERNEL_BUILDER(Name("BarrierOp").Device(DEVICE_CPU), BarrierOp);