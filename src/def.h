/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
enum OperationType{
    OPERATION_TYPE_TRAINING=0,
    OPERATION_TYPE_VALIDATION,
};

enum Layer{
    LAYER_TYPE_CONV=0,
    LAYER_TYPE_POOL,
    LAYER_TYPE_FULL,
    LAYER_TYPE_UPSAMPLE,
};

enum WorkerType{
    WORKER_TYPE_COMM=0,
    WORKER_TYPE_COMP,
};

enum CommunicationType{
    COMM_TYPE_REDUCE=0,
    COMM_TYPE_REDUCE_G,
    COMM_TYPE_REDUCE_P,
    COMM_TYPE_REDUCE_CONV_PARAM,
    COMM_TYPE_REDUCE_FULL_PARAM,
    COMM_TYPE_REDUCE_L,
    COMM_TYPE_GATHER_E,
    COMM_TYPE_GATHER_W,
    COMM_TYPE_GATHER_CONV_PARAM,
    COMM_TYPE_GATHER_M0,
    COMM_TYPE_GATHER_M1,
    COMM_TYPE_ALL2ALL_A,
    COMM_TYPE_ALL2ALL_G,
    COMM_TYPE_FINISH,
};

enum CheckStep{
    CHECK_STEP_REORDER=0,
    CHECK_STEP_PRINT,
    CHECK_STEP_DONE,
};

enum LossFunction{
    LOSS_TYPE_SOFTMAX=0,
    LOSS_TYPE_MAE,
    LOSS_TYPE_MSE,
    LOSS_TYPE_NONE,
};

enum ParamInitType{
    PARAM_INIT_TYPE_CONST=0,
    PARAM_INIT_TYPE_MSRA,
    PARAM_INIT_TYPE_GLOROT,
};

enum TaskType{
    TASK_TYPE_CLASSIFICATION=0,
    TASK_TYPE_REGRESSION,
};

enum Dataset{
    DATASET_MNIST=0,
    DATASET_CIFAR10,
    DATASET_IMAGENET,
    DATASET_PHANTOM,
    DATASET_DIV2K,
    DATASET_GHOST,
};

enum Optimizer{
    OPTIMIZER_SGD=0,
    OPTIMIZER_ADAM,
};