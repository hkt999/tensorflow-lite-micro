
# TensorFlow Lite Micro Library for Embedded System or MicroControllers

An Open Source Machine Learning Framework for Everyone. This project is derived from PICO tensorflow-lite project.
And make it buildable with Linux / MacOS and other toolchain, so that we can debug and test our models in a desktop
environment.

## Introduction

This is a version of the [TensorFlow Lite Micro library](https://www.tensorflow.org/lite/microcontrollers)
for the Linux OS. It allows you to run machine learning models to do things like voice recognition, detect people in
images, recognize gestures from an accelerometer, and other sensor analysis tasks.

## Getting Started

First you'll need to install toolchain on your system, include C/C++ compiler and CMake

You should then be able to build the library, tests, and examples. The easiest way to
build is using VS Code's CMake integration, by loading the project and choosing the
build option at the bottom of the window.

First you'll need to get the code from GIT, then you should then be able to build the library, tests, and examples.

## What's Included

There are several example applications included. The simplest one to begin with is the
hello_world project. This demonstrates the fundamentals of deploying an ML model on a
device, driving the Pico's LED in a learned sine-wave pattern.

Other examples include simple speech recognition, a magic wand gesture recognizer,
and spotting people in camera images, but because they require audio, accelerometer or
image inputs you'll need to write some code to hook up your own sensors, since these
are not included with the base microcontroller.

## Contributing

This repository is dereived from (https://github.com/raspberrypi/pico-tflmicro), and
make it as linux or MacOS built with changing the cmake files.

## Learning More

The [TensorFlow website](https://www.tensorflow.org/lite/microcontrollers) has
information on training, tutorials, and other resources.

The [TinyML Book](https://tinymlbook.com) is a guide to using TensorFlow Lite Micro
across a variety of different systems.

[TensorFlowLite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/pdf/2010.08678.pdf)
has more details on the design and implementation of the framework.

## Licensing

The TensorFlow source code is covered by the license described in src/tensorflow/LICENSE,
components from other libraries have the appropriate licenses included in their
third_party folders.

