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

import TensorFlow

// Original Paper:
// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
// Mingxing Tan, Quoc V. Le
// https://arxiv.org/abs/1905.11946
// Notes: Baseline (B0) network, table 1

public struct InitialMBConvBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var seReduceConv: Conv2D<Float>
    public var seExpandConv: Conv2D<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int)) {
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filters.0, 1),
            strides: (1, 1),
            padding: .same)
        seReduceConv = Conv2D<Float>(
            filterShape: (1, 1, filters.0, 8),
            strides: (1, 1),
            padding: .same)
        seExpandConv = Conv2D<Float>(
            filterShape: (1, 1, 8, filters.0),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, filters.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filters.0)
        batchNormConv = BatchNorm(featureCount: filters.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let dw = swish(batchNormDConv(dConv(input)))
        let se = sigmoid(seExpandConv(swish(seReduceConv(dw))))
        return conv2(se)
    }
}

public struct MBConvBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var seReduceConv: Conv2D<Float>
    public var seExpandConv: Conv2D<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        depthMultiplier: Int = 6,
        strides: (Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3)
    ) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        let hiddenDimension = filters.0 * depthMultiplier
        let reducedDimension = max(1, Int(filters.0 / 4))
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (kernel.0, kernel.1, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        seReduceConv = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, reducedDimension),
            strides: (1, 1),
            padding: .same)
        seExpandConv = Conv2D<Float>(
            filterShape: (1, 1, reducedDimension, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, hiddenDimension, filters.1),
            strides: (1, 1),
            padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount: filters.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let pw = swish(batchNormConv1(conv1(input)))
        var dw: Tensor<Float>
        if self.strides == (1, 1) {
            dw = swish(batchNormDConv(dConv(pw)))
        } else {
            dw = swish(zeroPad(batchNormDConv(dConv(pw))))
        }
        let se = sigmoid(seExpandConv(swish(seReduceConv(dw))))
        let pwLinear = batchNormConv2(conv2(se))

        if self.addResLayer {
            return input + pwLinear
        } else {
            return pwLinear
        }
    }
}

public struct MBConvBlockStack: Layer {
    var blocks: [MBConvBlock] = []

    public init(
        filters: (Int, Int),
        initialStrides: (Int, Int) = (2, 2),
        kernel: (Int, Int) = (3, 3),
        blockCount: Int
    ) {
        self.blocks = [MBConvBlock(filters: (filters.0, filters.1),
            strides: initialStrides, kernel: kernel)]
        for _ in 1..<blockCount {
            self.blocks.append(MBConvBlock(filters: (filters.1, filters.1), kernel: kernel))
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return blocks.differentiableReduce(input) { $1($0) }
    }
}

public struct EfficientNet: Layer {
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>
    public var initialMBConv = InitialMBConvBlock(filters: (32, 16))

    public var residualBlockStack1 = MBConvBlockStack(filters: (16, 24), blockCount: 2)
    public var residualBlockStack2 = MBConvBlockStack(filters: (24, 40), kernel: (5, 5),
        blockCount: 2)
    public var residualBlockStack3 = MBConvBlockStack(filters: (40, 80), blockCount: 3)
    public var residualBlockStack4 = MBConvBlockStack(filters: (80, 112), initialStrides: (1, 1),
        kernel: (5, 5), blockCount: 3)
    public var residualBlockStack5 = MBConvBlockStack(filters: (112, 192), kernel: (5, 5),
        blockCount: 4)
    public var residualBlockStack6 = MBConvBlockStack(filters: (192, 320), initialStrides: (1, 1),
        blockCount: 1)

    public var finalConv: Conv2D<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var dropout = Dropout<Float>(probability: 0.2)
    public var output: Dense<Float>

    public init(classCount: Int = 1000) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, 32),
            strides: (2, 2),
            padding: .same)
        inputConvBatchNorm = BatchNorm(featureCount: 32)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, 320, 1280),
            strides: (1, 1),
            padding: .same)
        output = Dense(inputSize: 1280, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: inputConv, inputConvBatchNorm,
            initialMBConv)
        let backbone = convolved.sequenced(through: residualBlockStack1, residualBlockStack2,
            residualBlockStack3, residualBlockStack4, residualBlockStack5, residualBlockStack6)
        return backbone.sequenced(through: finalConv, avgPool, dropout, output)
    }
}
