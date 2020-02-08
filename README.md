# Parallel Convolutional Neural Network (PCNN)

## Software Requirements
* OpenMP for parallelizing non-kernel operations based on the shared-memory programming model.
* MPI C and C++ compilers
* OpenCV for handling the input image files.
* Intel MKL for the kernel operations such as matrix-matrix multiplication.
* Boost for generating random numbers.

## Instructions to Build
1. Run the mklvar script first to set the MKL-related environment variables. We assume `MKLROOT` variable is set appropriately.
 ```
 source $HOME/intel/mkl/bin/mklvar.sh intel64
 ```
2. Generate configure file using autoreconf tool.
 ```
 autoreconf -i
 ```
3. Run 'configure' with the OpenCV path, for example
 ```
 ./configure --with-opencv=$HOME/lib/opencv
 ```
4. Run 'make' to create the executable 'pcnn'.

## Command to Run
* Command-line options are:
 ```
  mpiexec -n <np> ./pcnn [-s num] [-e num] [-t path] [-g num] [-i num]

  [-s]         enable training data shuffling (default: 0)
  [-e]         number of epochs to run (default: 1)
  [-t]         path to the checkpoint files
  [-g]         number of groups in local SGD training
  [-i]         model parameter averaging interval in local SGD training
 ```

## Supported CNN features
### Optimizers
 - SGD
 - Momentum SGD
 - Adam
 - localSGD

### Type of Model Layers
 - Convolution layers
 - Max/Avg-pooling layers
 - Fully-connected layers
 - Upsampling layers (super-resolution)
 
Note that PCNN supports only Rectified Linear Unit (ReLU) as an activation function.

### Type of Loss Functions
 - Softmax
 - MSE (L2)
 - MAE (L1)

### Other Optimization Features
 - Momentum
 - Learning rate decay
 - Residual connections
 - L2 regularization (a.k.a. weight decay)
 - Batch normalization (currently, only for convolution layers)

## Supported Parallelization Strategies
PCNN supports data-parallelism using two communication patterns:
 - Allreduce-based gradient averaging (traditional approach)
 - Multi-step communications for gradient averaging
 
## Supported Dataset
PCNN supports a couple of popular classification benchmark datasets: MNIST, CIFAR-10, and ImageNet, and a few regression dataset: DIV2K and Phantom. Ghost batch is the psuedo dataset that can be used for timing measurement. The proper flag and path variables should be set in config.h.

Example: CIFAR-10 dataset
1. Set #define CIFAR10 to 1 and all the other dataset flags to 0. 
2. Set #define CIFAR_TRAIN_TOP_DIR to the path where the training data is located.
3. Set #define CIFAR_TEST_TOP_DIR to the path where the test data is located.
4.  make clean; make

For other datasets, the path variables should be defined in a similar way.

## Predefined Model Architectures
PCNN supports a few popular model architectures in arch.c. The model should be chosen in config.h.
 - LeNet
 - ResNet20 / ResNet-50
 - VGG-16
 - DRRN
 - EDSR

## How to Setup Input Data
To use PCNN, the data feeding part should be implemented by users.
1. `DEPTH`, `WIDTH`, and `HEIGHT` of each training sample should be defined in config.h
2. `LABEL_SIZE` should be defined in config.h, which indicates the problem type (e.g., 10 for 10-class classification problems).
3. All the hyper-parameters should be specified in config.h, including mini-batch size, initial learning rate, and many others.
4. `TRAIN_TOP_DIR`, `TEST_TOP_DIR`, `TRAIN_LIST`, and `TEST_LIST` should be defined in config.h

Once all the dataset-dependent features are described in config.h, an appropriate data feeding function should be implemented.
By default, CIFAR10 and ImageNet data feeding functions are provided in feeder.c.

## Checkpointing
### Checkpoint Configurations
PCNN performs checkpointing based on the checkpoint configurations in config.h
In config.h, `CHECKPOINT_INTERVAL` indicates how frequently the model parameters are stored with respect to the number of epochs.
`CHECKPOINT_PATH` is the folder in which the checkpoint files will be stored.

### How to resume the training
1. Assuming the training algorithm is SGD, copy 'check-g0-xx.data' into the top folder after renaming it to 'check-g0.data'. Note that xx is the number of trained epochs.
2. Run PCNN with the `-t` option.
3. If the training algorithm is local SGD, repeat the step 1 for all the groups.

## Questions/Comments:
* Sunwoo Lee <slz839@eecs.northwestern.edu>
* Wei-keng Liao <wkliao@eecs.northwestern.edu>

## Project funding supports:
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program,
[RAPIDS Institute](https://rapids.lbl.gov).
