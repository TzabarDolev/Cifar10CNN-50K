# Cifar10CNN-50K
A CNN for the Cifar10 challenge which operates with 50K parameters, achieving 86.3% accuracy.

Deep learning: Convolutional networks exercise

created by Tzabar Dolev and Yotam Amitai (https://github.com/yotamitai)

The Model Architecture which obtained the best result for us was one with many
convolutional layers with an increasing number of channels. The last layer is fully
connected and reduces the number of neurons to the number of classes (10) and a Log
Soft-max function evaluates the label of a given image. The exact structure is described
below:
1. Convolution Layer: 3 to 16 channels
2. Convolution Layer: 16 to 16 channels
3. Convolution Layer: 16 to 16 channels + Batch Normalization + Max Pooling
4. Convolution Layer: 16 to 28 channels
5. Convolution Layer: 28 to 28 channels
6. Convolution Layer: 28 to 28 channels + Batch Normalization + Max-Pooling
7. Convolution Layer: 28 to 64 channels + Batch-Normalization + Max-Pooling +
Average-Pooling with a kernel of 1 × 1
8. Fully-Connected layer: 1024 to 10
9. Log Soft-max

Notes:
• All Convolutional kernels were of size 3 × 3 with a padding size of 1.

• All Convolutional layers included a PReLU activation function.

• All Max-Pooling kernels were of size 2 × 2

• Hyper-Parameters: Batch-size - 119, Learning-rate - 0.00089

• Optimizer - ADAM

• Augmentations- we tripled the data-set using three sets of augmentations:

1. RandomCrop, ColorJitterBrightness(0.2), ColorJitterSaturation(0.2), Normalize.

2. ColorJitterHue(0.2), ColorJitterContrast(0.2), RandomHorizontalFlip(1), Normalize.

3. RandomCrop, RandomAffine, Normalize.

• ACCURACY SCORE - 86.3%
