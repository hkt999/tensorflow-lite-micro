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

#include "main_functions.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.

  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 2 * 1024 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


char *read_model()
{
	int size;
	//FILE *infile = fopen("model/ssd_mobilenet_essay_neo.tflite", "rb");
	//FILE *infile = fopen("model/q_aware_model.tflite", "rb");
	//FILE *infile = fopen("model/q_aware_model-8-10.tflite", "rb");
	//FILE *infile = fopen("model/ssdlite_mobilenetv1_0.5_odapi.tflite", "rb");
	FILE *infile = fopen("model/ssdlite_mobilenetv1_0.5_int8_odapi.tflite", "rb");
	//FILE *infile = fopen("model/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite", "rb");
	if (infile == NULL) {
		printf("Error in opening model.\n");
		exit(0);
	}

	// get file size
	fseek(infile, 0, SEEK_END);
	size = ftell(infile);
	fseek(infile, 0, SEEK_SET);

	// allocate
	void *p = malloc(size);
	fread(p, 1, size, infile);
	fclose(infile);

	printf("read model p=%p, size=%d\n", p, size);
	return (char *) p;
}

void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  //model = tflite::GetModel(lite_model_ssd_mobilenet_v1_1_metadata_2_tflite);
  char *p = read_model();
  model = tflite::GetModel(p);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver all_ops_resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, all_ops_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  printf("Load model and allocate tensors success\n");

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  //input->data.uint8 = (uint8_t *)malloc(kMaxImageSize);
  input->data.uint8 = (uint8_t *)malloc(3*300*300);


  TfLiteTensor* output0 = interpreter->output(0);
  TfLiteTensor* output1 = interpreter->output(1);
  TfLiteTensor* output2 = interpreter->output(2);
  TfLiteTensor* output3 = interpreter->output(3);
  printf("output0: %p, output0->data.f: %p\n", output0, output0->data.f);
  printf("output1: %p, output1->data.f: %p\n", output1, output1->data.f);
  printf("output2: %p, output2->data.f: %p\n", output2, output2->data.f);
  printf("output3: %p, output3->data.f: %p\n", output3, output3->data.f);

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  printf("after involke !!\n");

  output0 = interpreter->output(0);
  output1 = interpreter->output(1);
  output2 = interpreter->output(2);
  output3 = interpreter->output(3);

  //printf("sizeof(*output0)=%ld\n", sizeof(*output0));
  //printf("sizeof(TfLiteTensor)=%ld\n", sizeof(TfLiteTensor));
  printf("output0: %p, output0->data.f: %p\n", output0, output0->data.f);
  printf("output1: %p, output1->data.f: %p\n", output1, output1->data.f);
  printf("output2: %p, output2->data.f: %p\n", output2, output2->data.f);
  printf("output3: %p, output3->data.f: %p\n", output3, output3->data.f);
  printf("%p, numbox=%d\n", output3->data.f, (int)output3->data.f[0]);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  exit(0);
}
