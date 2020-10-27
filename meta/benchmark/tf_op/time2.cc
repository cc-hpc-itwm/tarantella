#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Time2")
    .Input("input_tensor1: float")
    .Input("input_tensor2: float")
    .Output("out_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class Time2 : public OpKernel {
 public:
  explicit Time2(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    auto input1 = input_tensor1.flat<float>();
    auto input2 = input_tensor2.flat<float>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    const int N = input1.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = input1(i) + 2*input2(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Time2").Device(DEVICE_CPU), Time2);