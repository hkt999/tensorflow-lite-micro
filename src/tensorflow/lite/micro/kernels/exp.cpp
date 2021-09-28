/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/reference/exp.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace exp {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_EQ(context, input->dims->size, output->dims->size);
	  reference_ops::Exp(tflite::micro::GetTensorData<float>(input), input->dims->size, tflite::micro::GetTensorData<float>(output));
      break;

    default:
      TF_LITE_KERNEL_LOG(context, "Exp Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

} // namespace exp

TfLiteRegistration Register_EXP() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/exp::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
