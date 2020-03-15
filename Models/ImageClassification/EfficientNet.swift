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
// Notes: Default baseline (B0) network, see table 1

/// some utility functions to help generate network variants
/// original: https://github.com/tensorflow/tpu/blob/d6f2ef3edfeb4b1c2039b81014dc5271a7753832/models/official/efficientnet/efficientnet_model.py#L138
fileprivate var depthCoefficient: Float = 1.0
fileprivate func roundBlockDepthUp(blockCount: Int) -> Int {
    /// Multiply + round up the number of blocks based on global depth multiplier
    var newFilterCount = depthCoefficient * Float(blockCount)
    newFilterCount.round(.up)
    return Int(newFilterCount)
}

fileprivate var widthCoefficient: Float = 1.0
fileprivate func roundFilterCountDown(filter: Int, depthDivisor: Float = 8.0) -> Int {
    /// Multiply + round down the number of filters based on global width multiplier
    let filterMult = Float(filter) * widthCoefficient
    let filterAdd = Float(filterMult) + (depthDivisor / 2.0)
    var div = filterAdd / depthDivisor
    div.round(.down)
    div = div * Float(depthDivisor)
    var newFilterCount = max(1, Int(div))
    if newFilterCount < Int(0.9 * Float(filter)) {
        newFilterCount += Int(depthDivisor)
    }
    return Int(newFilterCount)
}

fileprivate func roundFilterPair(filters: (Int, Int)) -> (Int, Int) {
    return (roundFilterCountDown(filter: filters.0), roundFilterCountDown(filter: filters.1))
}

public struct InitialMBConvBlock: Layer {
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var seReduceConv: Conv2D<Float>
    public var seExpandConv: Conv2D<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int)) {
        let filterMult = roundFilterPair(filters: filters)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filterMult.0, 1),
            strides: (1, 1),
            padding: .same)
        seReduceConv = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, roundFilterCountDown(filter: 8)),
            strides: (1, 1),
            padding: .same)
        seExpandConv = Conv2D<Float>(
            filterShape: (1, 1, roundFilterCountDown(filter: 8), filterMult.0),
            strides: (1, 1),
            padding: .same)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filterMult.0)
        batchNormConv = BatchNorm(featureCount: filterMult.1)
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

        let filterMult = roundFilterPair(filters: filters)
        let hiddenDimension = filterMult.0 * depthMultiplier
        let reducedDimension = max(1, Int(filterMult.0 / 4))
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filterMult.0, hiddenDimension),
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
            filterShape: (1, 1, hiddenDimension, filterMult.1),
            strides: (1, 1),
            padding: .same)
        batchNormConv1 = BatchNorm(featureCount: hiddenDimension)
        batchNormDConv = BatchNorm(featureCount: hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount: filterMult.1)
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
        let blockMult = roundBlockDepthUp(blockCount: blockCount)
        self.blocks = [MBConvBlock(filters: (filters.0, filters.1),
            strides: initialStrides, kernel: kernel)]
        for _ in 1..<blockMult {
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
    public var initialMBConv: InitialMBConvBlock

    public var residualBlockStack1: MBConvBlockStack
    public var residualBlockStack2: MBConvBlockStack
    public var residualBlockStack3: MBConvBlockStack
    public var residualBlockStack4: MBConvBlockStack
    public var residualBlockStack5: MBConvBlockStack
    public var residualBlockStack6: MBConvBlockStack

    public var finalConv: Conv2D<Float>
    public var avgPool = GlobalAvgPool2D<Float>()
    public var dropoutProb: Dropout<Float>
    public var output: Dense<Float>

    /// default settings are efficientnetB0 (baseline) network
    /// resolution is here to show what the network can take as input, it doesn't set anything!
    public init(
        classCount: Int = 1000,
        width: Float = 1.0,
        depth: Float = 1.0,
        resolution: Int = 224,
        dropout: Double = 0.2
    ) {
        depthCoefficient = depth
        widthCoefficient = width

        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, roundFilterCountDown(filter: 32)),
            strides: (2, 2),
            padding: .same)
        inputConvBatchNorm = BatchNorm(featureCount: roundFilterCountDown(filter: 32))

        // global filter resizing happens at block layer
        initialMBConv = InitialMBConvBlock(filters: (32, 16))

        // global block resizing happens at stack layer
        residualBlockStack1 = MBConvBlockStack(filters: (16, 24), blockCount: 2)
        residualBlockStack2 = MBConvBlockStack(filters: (24, 40), kernel: (5, 5),
            blockCount: 2)
        residualBlockStack3 = MBConvBlockStack(filters: (40, 80), blockCount: 3)
        residualBlockStack4 = MBConvBlockStack(filters: (80, 112), initialStrides: (1, 1),
            kernel: (5, 5), blockCount: 3)
        residualBlockStack5 = MBConvBlockStack(filters: (112, 192), kernel: (5, 5),
            blockCount: 4)
        residualBlockStack6 = MBConvBlockStack(filters: (192, 320), initialStrides: (1, 1),
            blockCount: 1)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, roundFilterCountDown(filter: 320), roundFilterCountDown(filter: 1280)),
            strides: (1, 1),
            padding: .same)
        dropoutProb = Dropout<Float>(probability: dropout)
        output = Dense(inputSize: roundFilterCountDown(filter: 1280), outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: inputConv, inputConvBatchNorm,
            initialMBConv)
        let backbone = convolved.sequenced(through: residualBlockStack1, residualBlockStack2,
            residualBlockStack3, residualBlockStack4, residualBlockStack5, residualBlockStack6)
        return backbone.sequenced(through: finalConv, avgPool, dropoutProb, output)
    }
}

extension EfficientNet {
    public enum Kind {
        case efficientnetB0
        case efficientnetB1
        case efficientnetB2
        case efficientnetB3
        case efficientnetB4
        case efficientnetB5
        case efficientnetB6
        case efficientnetB7
        case efficientnetB8
    }

    public init(kind: Kind, classCount: Int = 1000) {
        switch kind {
        case .efficientnetB0:
            self.init(classCount: classCount, width: 1.0, depth: 1.0, resolution: 224, dropout: 0.2)
        case .efficientnetB1:
            self.init(classCount: classCount, width: 1.0, depth: 1.1, resolution: 240, dropout: 0.2)
        case .efficientnetB2:
            self.init(classCount: classCount, width: 1.1, depth: 1.2, resolution: 260, dropout: 0.3)
        case .efficientnetB3:
            self.init(classCount: classCount, width: 1.2, depth: 1.4, resolution: 300, dropout: 0.3)
        case .efficientnetB4:
            self.init(classCount: classCount, width: 1.4, depth: 1.8, resolution: 380, dropout: 0.4)
        case .efficientnetB5:
            self.init(classCount: classCount, width: 1.6, depth: 2.2, resolution: 456, dropout: 0.4)
        case .efficientnetB6:
            self.init(classCount: classCount, width: 1.8, depth: 2.6, resolution: 528, dropout: 0.5)
        case .efficientnetB7:
            self.init(classCount: classCount, width: 2.0, depth: 3.1, resolution: 600, dropout: 0.5)
        case .efficientnetB8:
            self.init(classCount: classCount, width: 2.2, depth: 3.6, resolution: 672, dropout: 0.5)
        }
    }
}
