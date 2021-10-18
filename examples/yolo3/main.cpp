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

#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "main_functions.h"
#include "hexdump.h"

using namespace cv;
using namespace std;
int main(int argc, char **argv)
{
	m_info_t info;

	if (argc < 3) {
		printf("Usage: %s [model] [photo]\n", argv[0]);
		return -1;
	}

	if (setup(argv[1], &info) != 0) {
		printf("error !\n");
		return -1;
	}

	printf("NN input width = %d\n", info.width);
	printf("NN input height = %d\n", info.height);
	Mat img, dstImg;
	img = imread(argv[2]);
	if (!img.data) {
		cout << "cannot open the image (" << argv[2] << ")" << std::endl;
		return -1;
	}
	resize(img, dstImg, Size(info.width, info.height));
	imshow("image", dstImg);
	cvtColor(dstImg, dstImg, COLOR_BGR2RGB);
	hexdump(dstImg.data, info.width);
	detect(dstImg.data);
	
	waitKey(0);

	return 0;
}
