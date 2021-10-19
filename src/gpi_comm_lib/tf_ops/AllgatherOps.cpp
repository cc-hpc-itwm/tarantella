#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "TensorAllgatherver.hpp"

#include <string.h>

using namespace tensorflow;

REGISTER_OP("AllgatherOp")
    .Attr("tnt_gatherer: int")
    .Input("input_tensor: float")
    .Input("output_shape: int32")
    .Output("output_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
    {
      c->set_output(0, c->input(1)); // set shape of output
      return Status::OK();
    });

template <typename Allgatherimplementation>
class AllgatherTensorOp : public OpKernel
{
  public:
    explicit AllgatherTensorOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
      tensorflow::int64 context_ptr;
      OP_REQUIRES_OK(context,
                    context->GetAttr("tnt_gatherer", &context_ptr));
      allgatherer = reinterpret_cast<tarantella::TensorAllgatherver *>(context_ptr);
    }

    void Compute(OpKernelContext* context) override
    {
      static_cast<Allgatherimplementation&>(*this).compute_impl(context);
    }

  protected:
    tarantella::TensorAllgatherver *allgatherer;
};

class AllgatherOp : public AllgatherTensorOp<AllgatherOp>
{
  public:
    using AllgatherTensorOp<AllgatherOp>::AllgatherTensorOp;

    void compute_impl(OpKernelContext* context)
    {
      const Tensor& input_tensor = context->input(0);
      auto* inputs = input_tensor.flat<float>().data();

      // construct shape using output size
      const Tensor& output_size_tensor = context->input(1);
      auto* output_size = output_size_tensor.flat<int>().data();

      TensorShape output_shape({*output_size, 1});

      //allocate output
      auto const output_index = 0;
      Tensor* output_tensor = nullptr;

      OP_REQUIRES_OK(context, context->allocate_output(output_index,
                                                       output_shape,
                                                       &output_tensor));
      auto* outputs = output_tensor->flat<float>().data();

      allgatherer->exec_allgatherv(inputs, outputs);
    }
};

REGISTER_KERNEL_BUILDER(Name("AllgatherOp").Device(DEVICE_CPU), AllgatherOp);