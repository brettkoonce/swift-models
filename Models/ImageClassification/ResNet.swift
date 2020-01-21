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
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// using shortcut layer to connect BasicBlock layers (aka Option (B))

public enum ImageSize {
    case cifar
    case imagenet
}

public struct ConvBN: Layer {
    public var conv: Conv2D<Float>
    public var norm: BatchNorm<Float>

    public init(
        filterShape: (Int, Int, Int, Int),
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.conv = Conv2D(filterShape: filterShape, strides: strides, padding: padding)
        self.norm = BatchNorm(featureCount: filterShape.3)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv, norm)
    }
}

public struct ResidualBasicBlockShortcut: Layer {
    public var layer1: ConvBN
    public var layer2: ConvBN
    public var shortcut: ConvBN

    public init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: (2, 2),
            padding: .same)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            strides: (1, 1),
            padding: .same)
        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: (2, 2),
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return layer2(relu(layer1(input))) + shortcut(input)
    }
}

public struct ResidualBasicBlock: Layer {
    public var layer1: ConvBN
    public var layer2: ConvBN

    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (1, 1)
    ) {
        self.layer1 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.0, featureCounts.1),
            strides: strides,
            padding: .same)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return layer2(relu(layer1(input)))
    }
}

public struct ResidualConvBlock: Layer {
    public var layer1: ConvBN
    public var layer2: ConvBN
    public var layer3: ConvBN
    public var shortcut: ConvBN

    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3,
        strides: (Int, Int) = (2, 2)
    ) {
        self.layer1 = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.1),
            strides: strides)
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)
        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
        self.shortcut = ConvBN(
            filterShape: (1, 1, featureCounts.0, featureCounts.3),
            strides: strides,
            padding: .same)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let tmp = relu(layer2(relu(layer1(input))))
        return relu(layer3(tmp) + shortcut(input))
    }
}

public struct ResidualIdentityBlock: Layer {
    public var layer1: ConvBN
    public var layer2: ConvBN
    public var layer3: ConvBN

    public init(featureCounts: (Int, Int, Int, Int), kernelSize: Int = 3) {
        self.layer1 = ConvBN(filterShape: (1, 1, featureCounts.0, featureCounts.1))
        self.layer2 = ConvBN(
            filterShape: (kernelSize, kernelSize, featureCounts.1, featureCounts.2),
            padding: .same)
        self.layer3 = ConvBN(filterShape: (1, 1, featureCounts.2, featureCounts.3))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let tmp = relu(layer2(relu(layer1(input))))
        return relu(layer3(tmp) + input)
    }
}

public struct ResidualBlockStack: Layer {
    public var basicBlocks: [ResidualBasicBlock] = []
    public var identityBlocks: [ResidualIdentityBlock] = []
    @noDerivative var isBasicBlockStack: Bool

    public init(basicType: Bool = false, featureCounts: (Int, Int, Int, Int),
        kernelSize: Int = 3, blockCount: Int) {
        self.isBasicBlockStack = basicType
        if basicType {
            for _ in 0..<blockCount {
                basicBlocks += [ResidualBasicBlock(featureCounts: featureCounts,
                    kernelSize: kernelSize)]
            }
        } else {
            for _ in 0..<blockCount {
                identityBlocks += [ResidualIdentityBlock(featureCounts: featureCounts,
                    kernelSize: kernelSize)]
            }
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        if self.isBasicBlockStack {
            let blocksReduced = basicBlocks.differentiableReduce(input) { last, layer in
                layer(last)
            }
            return blocksReduced
        } else {
            let blocksReduced = identityBlocks.differentiableReduce(input) { last, layer in
                layer(last)
            }
            return blocksReduced
        }
    }
}

public struct ResNet: Layer {
    @noDerivative var isBasicBlockStack: Bool

    public var l1: ConvBN
    public var maxPool: MaxPool2D<Float>

    public var l2a_bb = ResidualBasicBlock(featureCounts: (64, 64, 64, 64))
    public var l2a = ResidualConvBlock(featureCounts: (64, 64, 64, 256), strides: (1, 1))
    public var l2b: ResidualBlockStack

    public var l3a_bb = ResidualBasicBlockShortcut(featureCounts: (64, 128, 128, 128))
    public var l3a = ResidualConvBlock(featureCounts: (256, 128, 128, 512))
    public var l3b: ResidualBlockStack

    public var l4a_bb = ResidualBasicBlockShortcut(featureCounts: (128, 256, 256, 256))
    public var l4a = ResidualConvBlock(featureCounts: (512, 256, 256, 1024))
    public var l4b: ResidualBlockStack

    public var l5a_bb = ResidualBasicBlockShortcut(featureCounts: (256, 512, 512, 512))
    public var l5a = ResidualConvBlock(featureCounts: (1024, 512, 512, 2048))
    public var l5b: ResidualBlockStack

    public var avgPool = GlobalAvgPool2D<Float>()
    public var flatten = Flatten<Float>()
    public var classifier: Dense<Float>

    public init(basicType: Bool = false, classCount: Int, imageSize: ImageSize,
        layerBlockCounts: (Int, Int, Int, Int)) {
        self.isBasicBlockStack = basicType
        let bottleneckStackMult = basicType ? 1 : 4

        switch imageSize {
        case .imagenet:
            l1 = ConvBN(filterShape: (7, 7, 3, 64), strides: (2, 2), padding: .same)
            maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2))
        case .cifar:
            l1 = ConvBN(filterShape: (3, 3, 3, 64), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
        }

        l2b = ResidualBlockStack(basicType: basicType,
            featureCounts: (64 * bottleneckStackMult, 64, 64, 64 * bottleneckStackMult),
            blockCount: layerBlockCounts.0 - 1)
        l3b = ResidualBlockStack(basicType: basicType,
            featureCounts: (128 * bottleneckStackMult, 128, 128, 128 * bottleneckStackMult),
            blockCount: layerBlockCounts.1 - 1)
        l4b = ResidualBlockStack(basicType: basicType,
            featureCounts: (256 * bottleneckStackMult, 256, 256, 256 * bottleneckStackMult),
            blockCount: layerBlockCounts.2 - 1)
        l5b = ResidualBlockStack(basicType: basicType,
            featureCounts: (512 * bottleneckStackMult, 512, 512, 512 * bottleneckStackMult),
            blockCount: layerBlockCounts.3 - 1)
        classifier = Dense(inputSize: 512 * bottleneckStackMult, outputSize: classCount)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = maxPool(relu(l1(input)))
        if self.isBasicBlockStack {
            let level2 = inputLayer.sequenced(through: l2a_bb, l2b)
            let level3 = level2.sequenced(through: l3a_bb, l3b)
            let level4 = level3.sequenced(through: l4a_bb, l4b)
            let level5 = level4.sequenced(through: l5a_bb, l5b)
            return level5.sequenced(through: avgPool, flatten, classifier)
        } else {
            let level2 = inputLayer.sequenced(through: l2a, l2b)
            let level3 = level2.sequenced(through: l3a, l3b)
            let level4 = level3.sequenced(through: l4a, l4b)
            let level5 = level4.sequenced(through: l5a, l5b)
            return level5.sequenced(through: avgPool, flatten, classifier)
        }
    }
}

extension ResNet {
    public enum Depth {
        case resNet18
        case resNet34
        case resNet50
        case resNet101
        case resNet152
    }

    public init(classCount: Int, depth: Depth, imageSize: ImageSize) {
        switch depth {
        case .resNet18:
            self.init(basicType: true, classCount: classCount, imageSize: imageSize,
                layerBlockCounts: (2, 2, 2, 2))
        case .resNet34:
            self.init(basicType: true, classCount: classCount, imageSize: imageSize,
                layerBlockCounts: (3, 4, 6, 3))
        case .resNet50:
            self.init(basicType: false, classCount: classCount, imageSize: imageSize,
                layerBlockCounts: (3, 4, 6, 3))
        case .resNet101:
            self.init(basicType: false, classCount: classCount, imageSize: imageSize,
                layerBlockCounts: (3, 4, 23, 3))
        case .resNet152:
            self.init(basicType: false, classCount: classCount, imageSize: imageSize,
                layerBlockCounts: (3, 8, 36, 3))
        }
    }
}
