/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Time2")
    .Input("input1: float")
    .Input("input2: float")
    .Output("output: float");

void AddOneKernelLauncher(const float* in1,const float* in2, const int N, float* out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    auto input1 = input_tensor1.flat<float>();

    const Tensor& input_tensor2 = context->input(1);
    auto input2 = input_tensor2.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input1.size();
    // Call the cuda kernel launcher
    AddOneKernelLauncher(input1.data(),input2.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Time2").Device(DEVICE_GPU), AddOneOp);

