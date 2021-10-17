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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void ValidateExpGoldens(TfLiteTensor *tensors, int tensors_size,
		                const float* golden_exp, float *output_data) {

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = tflite::ops::micro::Register_EXP();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr, micro_test::reporter);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  int output_len = 1, size = tensors[0].dims->size;
  for (int i = 0; i < size; i++) {
    output_len *= tensors[0].dims->data[i];
  }
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ_FLOAT32(golden_exp[i], output_data[i]);
  }
}

void TestExpFloat(const int* input_dims_data, const float* input_data,
				  const float* golden_exp, float* output_data) {

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(input_dims_data); // same 

  TfLiteTensor input_tensor;
  input_tensor.type = TfLiteType::kTfLiteFloat32;
  input_tensor.dims = input_dims;
  input_tensor.data.f = (float *)input_data;

  TfLiteTensor output_tensor;
  output_tensor.type = TfLiteType::kTfLiteFloat32;
  output_tensor.dims = output_dims;
  output_tensor.data.f = (float *)output_data;

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
	  input_tensor,
	  output_tensor
  };
  ValidateExpGoldens(tensors, tensors_size, golden_exp, output_data);
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ExpOpTestFloat) {
  const int length = 10;
  const int dims[] = {2, 2, 5};
  const float values[] = { 0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5 };
  const float golden_values[] = { 1.000000, 4.481689, 7.389056, 12.182494, 20.085537,
		33.115452, 54.598148, 90.017128, 148.413162, 244.691925 };
  float output[length];

  tflite::testing::TestExpFloat( dims, values, golden_values, output);
}

TF_LITE_MICRO_TESTS_END
