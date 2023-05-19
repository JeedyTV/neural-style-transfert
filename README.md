# Neural Network Examples

This repository contains four files that demonstrate different neural network applications using the CIFAR-100 dataset. The files are organized as follows:

1. `train_vgg_cifar100.py`: This script trains a Convolutional Neural Network (CNN) using the VGG19 architecture on the CIFAR-100 dataset. It demonstrates how to preprocess the dataset, define the model architecture, train the model, and evaluate its performance.

2. `ourvgg.py`: This file defines the architecture of the CNN used in the `train_vgg19_cifar100.py` script. It provides the necessary functions and classes to construct the VGG19 model for image classification.

3. `background.py`: This script demonstrates how to separate the foreground and background of an image using neural networks. It takes an input image and utilizes a trained CNN to perform image segmentation, providing separate masks for the foreground and background.

4. `nst_vgg19.ipynb`: This file implements neural style transfer, a technique that combines the style of one image with the content of another. It uses a pre-trained CNN to extract features from the content and style images, and then generates a new image that preserves the content while adopting the style.

## Requirements

To run the scripts, you need the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras 2.x
- NumPy
- OpenCV
- Matplotlib

Please ensure that these dependencies are installed before running the scripts.

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/jeedytv/neural-style-transfert.git
   ```

2. Change into the repository directory:

   ```
   cd neural-style-transfert
   ```


   Note: The `train_vgg19_cifar100.py` script may take some time to train the CNN on the CIFAR-100 dataset, as it consists of a large number of images.

## Acknowledgments

- The CIFAR-100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- VGG19 architecture: https://arxiv.org/abs/1409.1556
- Neural style transfer: (https://arxiv.org/pdf/1508.06576.pdf)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to modify and distribute the code as per the terms of the license.

If you have any questions or encounter any issues, please don't hesitate to contact me.

Happy coding!
