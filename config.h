/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#define DEBUG 0
#define MAX_NUM_LAYERS 200
#define TEST_PER_EPOCH 1
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
/********** Parameter initialization **********
 * 0: Constant
 * Weight parameters are initialized with Gaussian N(0, std).
 * 1: Microsoft Research Asia (MSRA) method
 * Weight parameters are initialized with Gaussian N(0, sqrt(2/fan_in))
 * 2: Glorot normal method
 * Weight parameters are initialized with Gaussian N(0, sqrt(6/(fan_in + fan_out))) */
#define PARAM_INIT_METHOD 1
/********** Batch normalization parameters ****/
#define EPS_FACTOR 0.00001
#define MOVING_AVERAGE_FRACTION 0.999
/********** Data Pre-Processing ***************/
#define CROP_IMAGES 0
/********** Task type *************************
 * 0: CLASSIFICATION
 * The typical image classification task.
 * 1: REGRESSION
 * Cost minimization task. (e.g., image restoration) */
#define TASK_TYPE 0
#define UPSAMPLE_RATIO 2.0f
/********** Parallelization settings **********
 * COMM_PATTERN: 
 *     0: allreduce
 *     1: multi-step communications
 * OVERLAP
 *     0: no overlap
 *     1: overlap communication with computation */
#define COMM_PATTERN 1
#define OVERLAP 1
#define NUM_LAZY_LAYERS 8
/********** Model settings ********************/
#define MNIST_MODEL 0
#define CIFAR10_MODEL 0
#define VGGA 0
#define RESNET20 1 // for cifar10
#define RESNET50 0 // for imagenet
#define EDSR 0
#define DRRN 0
#define VDSR 0
/********** Checkpointing settings **********************/
#define CHECKPOINT_INTERVAL 1
#define CHECKPOINT_PATH "./checkpoints"
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

#define IMAGENET 0
#define IMAGENET_DEPTH 3
#define IMAGENET_WIDTH 224
#define IMAGENET_HEIGHT 224
#define IMAGENET_LABEL_SIZE 1000
#define IMAGENET_TRAIN_TOP_DIR "/raid/slz839/ImageNet/train"
#define IMAGENET_TRAIN_LIST "/raid/slz839/ImageNet/train.txt"
#define IMAGENET_TEST_TOP_DIR "/raid/slz839/ImageNet/valid"
#define IMAGENET_TEST_LIST "/raid/slz839/ImageNet/val.txt"

#define PHANTOM 0
#define PHANTOM_DEPTH 1
#define PHANTOM_ORIG_WIDTH 256
#define PHANTOM_ORIG_HEIGHT 256
#define PHANTOM_WIDTH 32
#define PHANTOM_HEIGHT 32
#define PHANTOM_LABEL_SIZE 1024
#define PHANTOM_TRAIN_TOP_DIR "/home/slz839/dataset/phantom_v2/train"
#define PHANTOM_TRAIN_LIST "/home/slz839/dataset/phantom_v2/train/list.txt"
#define PHANTOM_TEST_TOP_DIR "/home/slz839/dataset/phantom_v2/test"
#define PHANTOM_TEST_LIST "/home/slz839/dataset/phantom_v2/test/list.txt"

#define DIV2K 0
#define DIV2K_DEPTH 3
#define DIV2K_WIDTH 48
#define DIV2K_HEIGHT 48
#define DIV2K_LABEL_SIZE 27648
#define DIV2K_TRAIN_TOP_DIR "/home/slz839/dataset/div2k/train"
#define DIV2K_TRAIN_LIST "/home/slz839/dataset/div2k/train/list.txt"
#define DIV2K_TEST_TOP_DIR "/home/slz839/dataset/div2k/test"
#define DIV2K_TEST_LIST "/home/slz839/dataset/div2k/test/list.txt"

#define GHOST_BATCH 0
#define GHOST_BATCH_WIDTH 32
#define GHOST_BATCH_HEIGHT 32
#define GHOST_BATCH_DEPTH 3
#define GHOST_BATCH_LABEL_SIZE 3072
/**********************************************/
