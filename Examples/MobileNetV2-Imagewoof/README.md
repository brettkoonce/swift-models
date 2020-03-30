# MobileNet V2 with Imagewoof

This example demonstrates how to train the [MobileNetV2](https://arxiv.org/abs/1801.04381) network against the [Imagewoof image classification dataset](https://github.com/fastai/imagenette) (a harder version of Imagenette).

A Mobilenet V2 network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the Imagewoof dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

As a note: the current implementation of the Imagewoof dataset loads all images into memory as floats, which can lead to memory exhaustion on machines with less than 16 GB of available RAM.  Try reducing your batchSize if you are getting OOM errors.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release MobileNetV2-Imagewoof
```
