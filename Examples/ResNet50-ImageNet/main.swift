// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import ImageClassificationModels
import TensorBoard
import TensorFlow
import TrainingLoop
import Checkpoints
import Foundation

extension ResNet: Checkpointable {
  public var ignoredTensorPaths: Set<String> {
    return ["BatchNorm<Float>.runningMean", "BatchNorm<Float>.runningVariance"]
  }
}

// XLA mode can't load Imagenet, need to use eager mode to limit memory use
let device = Device.defaultTFEager
let dataset = ImageNet(batchSize: 32, outputSize: 224, on: device)
var model = ResNet(classCount: 1000, depth: .resNet50)

let offset = 0
let totalEpochs = 90
let temporaryDirectory = URL(fileURLWithPath: "/tmp/resnet50-imagenet/\(offset)/resnet50/")

var readCorrectly = false
print ("reading model (epoch: \(offset)) from tmp directory...")
do {
  try model.readCheckpoint(from: temporaryDirectory.appendingPathComponent("ResNet"), name: "ResNet")
  print ("success reading model (epoch: \(offset))!")
  readCorrectly = true
} catch {
  print(error)
}

if (!readCorrectly && offset != 0) {
  print ("couldn't find desired epoch to load, halting!")
  exit(0)
}

// https://github.com/mlcommons/training/blob/4f97c909f3aeaa3351da473d12eba461ace0be76/image_classification/tensorflow/official/resnet/imagenet_main.py#L286
let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)
public func scheduleLearningRate<L: TrainingLoopProtocol>(
  _ loop: inout L, event: TrainingLoopEvent
) throws where L.Opt.Scalar == Float {
  if event == .epochStart {
    guard let epoch = loop.epochIndex else  { return }
    if epoch + offset > 30 { loop.optimizer.learningRate = 0.01 }
    if epoch + offset > 60 { loop.optimizer.learningRate = 0.001 }
    if epoch + offset > 80 { loop.optimizer.learningRate = 0.0001 }
  }
}

func saveCallback<L: TrainingLoopProtocol>(
  _ loop: inout L, event: TrainingLoopEvent
) throws where L.Opt.Scalar == Float {
  if event == .epochEnd {
    guard let epoch = loop.epochIndex else  { return }
    let epochOut = offset + epoch + 1
    let temporaryDirectory = URL(fileURLWithPath: "/tmp/resnet50-imagenet/\(epochOut)/resnet50/")

    DispatchQueue.global(qos: .userInitiated).async {
      print("\nsaving model: \(epochOut)")
      do {
        try model.writeCheckpoint(to: temporaryDirectory, name: "ResNet")
      } catch {
        print(error)
      }
      print("\nsaved model: \(epochOut)")
    }
  }
}

var trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy],
  callbacks: [scheduleLearningRate, tensorBoardStatisticsLogger(), saveCallback])

try! trainingLoop.fit(&model, epochs: totalEpochs - offset, on: device)

print ("waiting for last model write...")
sleep(10)
