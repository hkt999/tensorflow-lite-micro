/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace transpose {

// This file has two implementations of Transpose.
enum KernelType {
  kReference,
  kGenericOptimized,
};
struct TransposeContext {
  TransposeContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    perm = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* perm;
  TfLiteTensor* output;
};

#ifndef TF_LITE_STATIC_MEMORY
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                TransposeContext* op_context) {
  int dims = NumDimensions(op_context->input);
  const int* perm_data = GetTensorData<int32_t>(op_context->perm);

  // Ensure validity of the permutations tensor as a 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->perm), 1);
  TF_LITE_ENSURE_EQ(context, op_context->perm->dims->data[0], dims);
  for (int idx = 0; idx < dims; ++idx) {
    TF_LITE_ENSURE_MSG(context, (perm_data[idx] >= 0 && perm_data[idx] < dims),
                       "Transpose op permutations array is out of bounds.");
  }

  // Determine size of output tensor.
  TfLiteIntArray* input_size = op_context->input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
  for (int idx = 0; idx < dims; ++idx) {
    output_size->data[idx] = input_size->data[perm_data[idx]];
  }

  return context->ResizeTensor(context, op_context->output, output_size);
}
#endif

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TransposeContext op_context(context, node);

  // Ensure validity of input tensor.
  TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                     "Transpose op only supports 1D-5D input arrays.");
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);

  if (!IsConstantTensor(op_context.perm)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
#ifndef TF_LITE_STATIC_MEMORY
  return ResizeOutputTensor(context, &op_context);
#else
  return kTfLiteError;
#endif
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

#if 0
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }
#endif

  const int* perm_data = GetTensorData<int32_t>(op_context.perm);
  const int size = op_context.perm->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

#define TF_LITE_TRANSPOSE(type, scalar)                     \
  type::Transpose(params, GetTensorShape(op_context.input), \
                  GetTensorData<scalar>(op_context.input),  \
                  GetTensorShape(op_context.output),        \
                  GetTensorData<scalar>(op_context.output))

  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (op_context.input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
        TF_LITE_TRANSPOSE(reference_ops, int32_t);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
        TF_LITE_TRANSPOSE(reference_ops, int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_TRANSPOSE(reference_ops, int16_t);
      break;
    case kTfLiteInt64:
      TF_LITE_TRANSPOSE(reference_ops, int64_t);
      break;
    case kTfLiteBool:
      if (sizeof(bool) == 1) {
        TF_LITE_TRANSPOSE(reference_ops, int8_t);
      } else {
        TF_LITE_TRANSPOSE(reference_ops, bool);
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported by Transpose.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_TRANSPOSE

  return kTfLiteOk;
}

}  // namespace transpose

TfLiteRegistration* Register_TRANSPOSE_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, transpose::Prepare,
                                 transpose::Eval<transpose::kReference>};
  return &r;
}

#if 0
TfLiteRegistration* Register_TRANSPOSE_GENERIC_OPTIMIZED() {
  static TfLiteRegistration r = {nullptr, nullptr, transpose::Prepare,
                                 transpose::Eval<transpose::kGenericOptimized>};
  return &r;
}
#endif

TfLiteRegistration* Register_TRANSPOSE() {
  return Register_TRANSPOSE_REF();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
