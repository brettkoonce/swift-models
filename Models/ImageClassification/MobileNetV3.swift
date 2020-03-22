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

// Original Paper: "Searching for MobileNetV3"
// Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang,
// Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
// https://arxiv.org/abs/1905.02244

public struct InitialInvertedResidualBlock: Layer {
    @noDerivative public var addResLayer: Bool
    @noDerivative public var useSELayer: Bool = false
    @noDerivative public var useHardSwish: Bool = false
    @noDerivative public var hiddenDimension: Int

    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var seAveragePool = GlobalAvgPool2D<Float>()
    public var seReduceConv: Dense<Float>
    public var seExpandConv: Dense<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        strides: (Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3),
        seLayer: Bool = false,
        hSwish: Bool = false
    ) {
        self.useSELayer = seLayer
        self.useHardSwish = hSwish
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)

        self.hiddenDimension = filters.0
        let reducedDimension = max(1, Int(filters.0 / 4))

        dConv = DepthwiseConv2D<Float>(
            filterShape: (3, 3, filters.0, 1),
            strides: (1, 1),
            padding: .same)
        seReduceConv = Dense(inputSize: hiddenDimension, outputSize: reducedDimension)
        seExpandConv = Dense(inputSize: reducedDimension, outputSize: hiddenDimension)
        conv2 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, filters.1),
            strides: (1, 1),
            padding: .same)
        batchNormDConv = BatchNorm(featureCount: filters.0)
        batchNormConv2 = BatchNorm(featureCount: filters.1)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var depthwise = batchNormDConv(dConv(input))
        if self.useHardSwish {
            depthwise = hardSwish(depthwise)
        } else {
            depthwise = relu(depthwise)
        }
        var squeezeExcite: Tensor<Float>
        if self.useSELayer {
            let seAvgPoolReshaped = seAveragePool(depthwise).reshaped(to: [
                input.shape[0], 1, 1, self.hiddenDimension
            ])
            squeezeExcite = depthwise
                * sigmoid(seExpandConv(relu(seReduceConv(seAvgPoolReshaped))))
        } else {
            squeezeExcite = depthwise
        }

        let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

        if self.addResLayer {
            return input + piecewiseLinear
        } else {
            return piecewiseLinear
        }
    }
}

public struct InvertedResidualBlock: Layer {
    @noDerivative public var strides: (Int, Int)
    @noDerivative public let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    @noDerivative public var hiddenDimension: Int
    @noDerivative public var addResLayer: Bool
    @noDerivative public var useHardSwish: Bool
    @noDerivative public var useSELayer: Bool

    public var conv1: Conv2D<Float>
    public var batchNormConv1: BatchNorm<Float>
    public var dConv: DepthwiseConv2D<Float>
    public var batchNormDConv: BatchNorm<Float>
    public var seAveragePool = GlobalAvgPool2D<Float>()
    public var seReduceConv: Dense<Float>
    public var seExpandConv: Dense<Float>
    public var conv2: Conv2D<Float>
    public var batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        hiddenDimension: Int,
        strides: (Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3),
        seLayer: Bool = false,
        hSwish: Bool = false
    ) {
        self.strides = strides
        self.addResLayer = filters.0 == filters.1 && strides == (1, 1)
        self.useSELayer = seLayer
        self.useHardSwish = hSwish
        self.hiddenDimension = hiddenDimension

        let reducedDimension = hiddenDimension / 4
        conv1 = Conv2D<Float>(
            filterShape: (1, 1, filters.0, hiddenDimension),
            strides: (1, 1),
            padding: .same)
        dConv = DepthwiseConv2D<Float>(
            filterShape: (kernel.0, kernel.1, hiddenDimension, 1),
            strides: strides,
            padding: strides == (1, 1) ? .same : .valid)
        seReduceConv = Dense(inputSize: hiddenDimension, outputSize: reducedDimension)
        seExpandConv = Dense(inputSize: reducedDimension, outputSize: hiddenDimension)
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
        var piecewise = batchNormConv1(conv1(input))
        if self.useHardSwish {
            piecewise = hardSwish(piecewise)
        } else {
            piecewise = relu(piecewise)
        }
        var depthwise: Tensor<Float>
        if self.strides == (1, 1) {
            depthwise = batchNormDConv(dConv(piecewise))
        } else {
            depthwise = batchNormDConv(dConv(zeroPad(piecewise)))
        }
        if self.useHardSwish {
            depthwise = hardSwish(depthwise)
        } else {
            depthwise = relu(depthwise)
        }
        var squeezeExcite: Tensor<Float>
        if self.useSELayer {
            let seAvgPoolReshaped = seAveragePool(depthwise).reshaped(to: [
                input.shape[0], 1, 1, self.hiddenDimension
            ])
            squeezeExcite = depthwise
                * sigmoid(seExpandConv(relu(seReduceConv(seAvgPoolReshaped))))
        } else {
            squeezeExcite = depthwise
        }

        let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

        if self.addResLayer {
            return input + piecewiseLinear
        } else {
            return piecewiseLinear
        }
    }
}

public struct MobileNetV3Large: Layer {
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>

    public var invertedResidualBlock1 = InitialInvertedResidualBlock(filters: (16, 16))
    public var invertedResidualBlock2 = InvertedResidualBlock(
        filters: (16, 24),
        hiddenDimension: 64, strides: (2, 2))
    public var invertedResidualBlock3 = InvertedResidualBlock(
        filters: (24, 24),
        hiddenDimension: 72)
    public var invertedResidualBlock4 = InvertedResidualBlock(
        filters: (24, 40),
        hiddenDimension: 72, strides: (2, 2), kernel: (5, 5), seLayer: true)
    public var invertedResidualBlock5 = InvertedResidualBlock(
        filters: (40, 40),
        hiddenDimension: 120, kernel: (5, 5), seLayer: true)
    public var invertedResidualBlock6 = InvertedResidualBlock(
        filters: (40, 40),
        hiddenDimension: 120, kernel: (5, 5), seLayer: true)
    public var invertedResidualBlock7 = InvertedResidualBlock(
        filters: (40, 80),
        hiddenDimension: 240, strides: (2, 2), hSwish: true)
    public var invertedResidualBlock8 = InvertedResidualBlock(
        filters: (80, 80),
        hiddenDimension: 200, hSwish: true)
    public var invertedResidualBlock9 = InvertedResidualBlock(
        filters: (80, 80),
        hiddenDimension: 184, hSwish: true)
    public var invertedResidualBlock10 = InvertedResidualBlock(
        filters: (80, 80),
        hiddenDimension: 184, hSwish: true)
    public var invertedResidualBlock11 = InvertedResidualBlock(
        filters: (80, 112),
        hiddenDimension: 480, seLayer: true, hSwish: true)
    public var invertedResidualBlock12 = InvertedResidualBlock(
        filters: (112, 112),
        hiddenDimension: 672, seLayer: true, hSwish: true)
    public var invertedResidualBlock13 = InvertedResidualBlock(
        filters: (112, 160),
        hiddenDimension: 672, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock14 = InvertedResidualBlock(
        filters: (160, 160),
        hiddenDimension: 672, strides: (2, 2), kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock15 = InvertedResidualBlock(
        filters: (160, 160),
        hiddenDimension: 960, kernel: (5, 5), seLayer: true, hSwish: true)

    public var outputConv: Conv2D<Float>
    public var outputConvBatchNorm: BatchNorm<Float>

    public var avgPool = GlobalAvgPool2D<Float>()
    public var finalConv: Conv2D<Float>
    public var classiferConv: Conv2D<Float>
    public var flatten = Flatten<Float>()

    public init(classCount: Int = 1000) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, 16),
            strides: (2, 2),
            padding: .same)
        inputConvBatchNorm = BatchNorm(featureCount: 16)

        outputConv = Conv2D<Float>(
            filterShape: (1, 1, 160, 960),
            strides: (1, 1),
            padding: .same)
        outputConvBatchNorm = BatchNorm(featureCount: 960)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, 960, 1280),
            strides: (1, 1),
            padding: .same)
        classiferConv = Conv2D<Float>(
            filterShape: (1, 1, 1280, classCount),
            strides: (1, 1),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let initialConv = hardSwish(input.sequenced(through: inputConv, inputConvBatchNorm))
        let backbone1 = initialConv.sequenced(
            through: invertedResidualBlock1,
            invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
            invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(
            through: invertedResidualBlock6, invertedResidualBlock7,
            invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10)
        let backbone3 = backbone2.sequenced(
            through: invertedResidualBlock11,
            invertedResidualBlock12, invertedResidualBlock13, invertedResidualBlock14,
            invertedResidualBlock15)
        let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone3)))
        let averagePool = avgPool(outputConvResult).reshaped(to: [
            input.shape[0], 1, 1, 960
        ])
        let finalConvResult = hardSwish(finalConv(averagePool))
        return softmax(flatten(classiferConv(finalConvResult)))
    }
}

public struct MobileNetV3Small: Layer {
    public var inputConv: Conv2D<Float>
    public var inputConvBatchNorm: BatchNorm<Float>

    public var invertedResidualBlock1 = InitialInvertedResidualBlock(
        filters: (16, 16),
        strides: (2, 2), seLayer: true)
    public var invertedResidualBlock2 = InvertedResidualBlock(
        filters: (16, 24),
        hiddenDimension: 72, strides: (2, 2))
    public var invertedResidualBlock3 = InvertedResidualBlock(
        filters: (24, 24),
        hiddenDimension: 88)
    public var invertedResidualBlock4 = InvertedResidualBlock(
        filters: (24, 40),
        hiddenDimension: 96, strides: (2, 2), kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock5 = InvertedResidualBlock(
        filters: (40, 40),
        hiddenDimension: 240, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock6 = InvertedResidualBlock(
        filters: (40, 40),
        hiddenDimension: 240, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock7 = InvertedResidualBlock(
        filters: (40, 48),
        hiddenDimension: 120, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock8 = InvertedResidualBlock(
        filters: (48, 48),
        hiddenDimension: 144, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock9 = InvertedResidualBlock(
        filters: (48, 96),
        hiddenDimension: 288, strides: (2, 2), kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock10 = InvertedResidualBlock(
        filters: (96, 96),
        hiddenDimension: 576, kernel: (5, 5), seLayer: true, hSwish: true)
    public var invertedResidualBlock11 = InvertedResidualBlock(
        filters: (96, 96),
        hiddenDimension: 576, kernel: (5, 5), seLayer: true, hSwish: true)

    public var outputConv: Conv2D<Float>
    public var outputConvBatchNorm: BatchNorm<Float>

    public var avgPool = GlobalAvgPool2D<Float>()
    public var finalConv: Conv2D<Float>
    public var classiferConv: Conv2D<Float>
    public var flatten = Flatten<Float>()

    public init(classCount: Int = 1000) {
        inputConv = Conv2D<Float>(
            filterShape: (3, 3, 3, 16),
            strides: (2, 2),
            padding: .same)
        inputConvBatchNorm = BatchNorm(featureCount: 16)

        outputConv = Conv2D<Float>(
            filterShape: (1, 1, 96, 576),
            strides: (1, 1),
            padding: .same)
        outputConvBatchNorm = BatchNorm(featureCount: 576)

        finalConv = Conv2D<Float>(
            filterShape: (1, 1, 576, 1280),
            strides: (1, 1),
            padding: .same)
        classiferConv = Conv2D<Float>(
            filterShape: (1, 1, 1280, classCount),
            strides: (1, 1),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let initialConv = hardSwish(input.sequenced(through: inputConv, inputConvBatchNorm))
        let backbone1 = initialConv.sequenced(
            through: invertedResidualBlock1,
            invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
            invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(
            through: invertedResidualBlock6, invertedResidualBlock7,
            invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10,
            invertedResidualBlock11)
        let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone2)))
        let averagePool = avgPool(outputConvResult).reshaped(to: [
            input.shape[0], 1, 1, 576
        ])
        let finalConvResult = hardSwish(finalConv(averagePool))
        return softmax(flatten(classiferConv(finalConvResult)))
    }
}
