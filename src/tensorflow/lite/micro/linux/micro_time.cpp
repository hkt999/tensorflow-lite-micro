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

// Raspberry Pi Pico-specific implementation of timing functions.

#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/debug_log.h"
#include <sys/time.h>

namespace tflite {

int32_t ticks_per_second() { return 1000; }

static struct timeval start_tv;
int32_t GetCurrentTimeTicks() {
	static int started = 0;
	if ( started == 0 ) {
		gettimeofday(&start_tv, 0);
		started = 1;
	}
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (int32_t)((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
			(start_tv.tv_sec * 1000 + start_tv.tv_usec / 1000));
}

}  // namespace tflite
