#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "op_utils.hpp"

#include "TensorAllgatherver.hpp"

using namespace tensorflow;

REGISTER_OP("AllgatherOp")
    .Attr("tnt_gatherer: int")
    .Attr("batch_size: int")
    .Input("input_tensor: float")
    .Output("output_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
    {
      TFLongIntType batch_size;
      c->GetAttr("batch_size", &batch_size);

      shape_inference::ShapeHandle sample_shape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &sample_shape));

      shape_inference::ShapeHandle batch_shape = c->MakeShape({batch_size});
      shape_inference::ShapeHandle out_shape;

      TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, sample_shape, &out_shape));
      c->set_output(0, out_shape);
      return Status::OK();
    });

class AllgatherOp : public OpKernel
{
  public:
    explicit AllgatherOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
      TFLongIntType context_ptr;
      OP_REQUIRES_OK(context,
                    context->GetAttr("tnt_gatherer", &context_ptr));
      allgatherer = reinterpret_cast<tarantella::TensorAllgatherver *>(context_ptr);
    }

    void Compute(OpKernelContext* context)
    {
      const Tensor& input_tensor = context->input(0);
      auto* inputs = input_tensor.flat<float>().data();

      // construct shape using output size
      TFLongIntType size = allgatherer->getOutputCount();
      TensorShape output_shape{size};

      //allocate output
      auto const output_index = 0;
      Tensor* output_tensor = nullptr;

      OP_REQUIRES_OK(context, context->allocate_output(output_index,
                                                       output_shape,
                                                       &output_tensor));
      auto* outputs = output_tensor->flat<float>().data();
      allgatherer->exec_allgatherv(inputs, outputs);
    }

  private:
    tarantella::TensorAllgatherver *allgatherer;
};

REGISTER_KERNEL_BUILDER(Name("AllgatherOp").Device(DEVICE_CPU), AllgatherOp);
