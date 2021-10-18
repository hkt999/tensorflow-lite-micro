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
#include <stdlib.h>

#define DEBUG0(fmt...)	printf(fmt)

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
  // constexpr int kTensorArenaSize = 1.2 * 1024 * 1024;
  constexpr int kTensorArenaSize = 4 * 1024 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


char *read_model(char *model_filename)
{
	int size;
	FILE *infile = fopen(model_filename, "rb");
	if (infile == NULL) {
		printf("Error in opening model (%s).\n", model_filename);
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


static void dump_tensor(const char *name, TfLiteTensor *tensor)
{
  DEBUG0("%s: %p, data.raw: %p, dims->size=%d, type=%d, size=%ld\n", name,
    tensor, tensor->data.f, tensor->dims->size, tensor->type, tensor->bytes);
  for (int i=0; i<tensor->dims->size; i++) {
	  DEBUG0("  dim(%d)=%d\n", i, tensor->dims->data[i]);
  }
}

int setup(char *model_filename, m_info_t *info) 
{
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  //model = tflite::GetModel(lite_model_ssd_mobilenet_v1_1_metadata_2_tflite);
  model = tflite::GetModel(read_model(model_filename));
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  static tflite::AllOpsResolver all_ops_resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, all_ops_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  DEBUG0("interpreter->AllocateTensors()\n");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

#if 0
  static float _bounding_box[4*10];
  static float _score[10];
  static float _class[10];

  // rewrite the output data for default post processor 
  TfLiteTensor* output0 = interpreter->output(0);
  TfLiteTensor* output1 = interpreter->output(1);
  TfLiteTensor* output2 = interpreter->output(2);

  DEBUG0("before output0->data.f=%p\n", output0->data.f);
  output0->data.f = (float *)_bounding_box;
  output0->bytes = sizeof(_bounding_box);
  DEBUG0("after output0->data.f=%p\n", output0->data.f);
  output1->data.f = (float *)_class;
  output1->bytes = sizeof(_class);
  output2->data.f = (float *)_score;
  output2->bytes = sizeof(_score);
#endif

  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return 1;
  }

  memset(info, 0, sizeof(m_info_t));
  DEBUG0("Load model and allocate tensors success\n");
  input = interpreter->input(0);
  dump_tensor("input_tensor", input);
  info->bytes = input->bytes;
  info->width = input->dims->data[2];
  info->height = input->dims->data[1];

  return 0;
}

void detect(unsigned char *image)
{
  int8_t *dst = (int8_t *)input->data.int8;
  uint8_t *src = (uint8_t *)image;
  int count = input->bytes;
  while (count-->0) {
	  *dst++ = (int8_t)((int)(*src++) - 128);
  }
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output0 = interpreter->output(0);
  TfLiteTensor* output1 = interpreter->output(1);
  dump_tensor("output0", output0);
  dump_tensor("output1", output1);

  float *boxes = output0->data.f;
  float *scores = output1->data.f;
  int bboxes_num = output0->dims->data[1];
  DEBUG0("number of box = %d\n", bboxes_num);
  int classes_num = 1;
  int score_threshold = 5;
  int detected_objects = 0;
  for (int i=0;i<bboxes_num;i++) {
      float score = scores[i];
      if (int(score * 100) > score_threshold ) {
        detected_objects++;
        float x = boxes[i*4];
        float y = boxes[i*4+1];
        float w = boxes[i*4+2];
        float h = boxes[i*4+3];
        DEBUG0("inference raw output score=%f (cx=%f, cy=%f), (w=%f, h=%f)\n", score, x, y, w, h);
#if 0
        int minx = (x - w / 2.0);
        int maxx = (x + w / 2.0);
        int miny = (y - h / 2.0);
        int maxy = (y + h / 2.0);
        DEBUG0("score: %d, box: (%d,%d), (%d, %d)\n", (int)(score* 100), minx, miny, maxx, maxy);
#endif

        // BoundingBox box( 1, 1, 100, 100, score * 100, 0);
        // nms.AddBoundingBox(box);
        //if (detect_cb != NULL) {
        //  detect_cb(0, int(score * 100), 0, 0, 0, 0);
        //}
      }
  }

#if 0 // mobilenet ssd postprocess output
  #define COL_RES	416
  #define ROW_RES	234
  int numBoxes = 1365;
  DEBUG0("numBoxes=%d\n", numBoxes);
  for (int i=0; i<numBoxes; i++) {
    int minX = (int)(COL_RES * out0[i*4+1]);
    int minY = (int)(ROW_RES * out0[i*4]);
    int maxX = (int)(COL_RES * out0[i*4+3]);
    int maxY = (int)(ROW_RES * out0[i*4+2]);
    int score = (int)(out1[i] * 100.0);
	printf("idx(%d) score=%d, (%d,%d) - (%d,%d)\n",i, score, minX, minY, maxX, maxY);
  }
#endif
}

