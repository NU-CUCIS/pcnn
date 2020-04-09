# Classification with PCNN
To solve classfication problems with PCNN, users should define the model architecture and specify the hyper-parameters.
By default, PCNN supports a few open benchmark datasets, MNIST, CIFAR10, and ImageNet. 
Let us use CIFAR10 as an example.
CIFAR10 is a 10-class classification dataset that is publicly open.
It consists of 3-channel (RGB) 50,000 training images of size 32 x 32 and 10,000 validation images.
The dataset can be obtained [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Getting Started
* To train a model on CIFAR10 with PCNN, first users should define a model in `arch.c`. PCNN supports ResNet20 as the default model architecture for CIFAR10 classification.
* Then, specify the dataset in `config.h` as follows.
```
/********** Model settings ********************/
#define MNIST_MODEL 0
#define CIFAR10_MODEL 0
#define VGGA 0
#define RESNET20 1 // for cifar10
#define RESNET50 0 // for imagenet
#define EDSR 0
#define DRRN 0
#define VDSR 0
```

* Specify the task type to classification in `config.h`.
```
/********** Task type *************************
 * 0: CLASSIFICATION
 * The typical image classification task.
 * 1: REGRESSION
 * Cost minimization task. (e.g., image restoration) */
#define TASK_TYPE 0
```

* Specify the dataset in `config.h` as follows (#define CIFAR10 1).
```
/********** Dataset settings ******************/
#define MNIST 0
#define MNIST_DEPTH 1
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define MNIST_LABEL_SIZE 10
#define MNIST_TRAIN_TOP_DIR "/home/slz839/dataset/mnist"
#define MNIST_TEST_TOP_DIR "/home/slz839/dataset/mnist"
#define MNIST_TRAIN_IMAGE "train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL "train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGE "t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL "t10k-labels-idx1-ubyte"

#define CIFAR10 1
#define CIFAR10_DEPTH 3
#define CIFAR10_WIDTH 32
#define CIFAR10_HEIGHT 32
#define CIFAR10_LABEL_SIZE 10
#define CIFAR10_TRAIN_TOP_DIR "/home/slz839/dataset/cifar10"
#define CIFAR10_TEST_TOP_DIR "/home/slz839/dataset/cifar10"
#define CIFAR10_TRAIN_IMAGE1 "data_batch_1.bin"
#define CIFAR10_TRAIN_IMAGE2 "data_batch_2.bin"
#define CIFAR10_TRAIN_IMAGE3 "data_batch_3.bin"
#define CIFAR10_TRAIN_IMAGE4 "data_batch_4.bin"
#define CIFAR10_TRAIN_IMAGE5 "data_batch_5.bin"
#define CIFAR10_TEST_IMAGE "test_batch.bin"
```

* Check all the hyper-parameters defined in `config.h`.
```
/********** Optimizer *************************
 * 0: Mini-batch SGD
 * 1: Adam */
#define OPTIMIZER 0
/********** Optimizer-dependent settings ******/
#define WEIGHT_DECAY 0.0001
#define MOMENTUM 0.9
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 1e-8
/********** Hyper-parameter settings **********/
#define BATCH_SIZE 128
#define LEARNING_RATE 0.1
#define LEARNING_RATE_DECAY_STEPS 80
#define LEARNING_RATE_DECAY_FACTOR 0.1f
```

## Running PCNN
Once all the configurations are appropriately set and PCNN is successfully built, the training can be started as follows.
For building the code and input arguments, please refer to the README in `src` directory.
```
mpiexec -n 4 -f nodes ./pcnn -s 1 -e 160
```

## Training Results
