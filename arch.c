/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "def.h"
#include "model.h"

/* Arch module is for users to define the model architecture as a function.
 * A few popular models are already defined here,
 * such as lenet, VGG16, resnet20/50, and EDSR. */
void pcnn_arch_config_lenet(struct model_t *model, struct feeder_t *feeder)
{
    struct feature_t features;

    pcnn_model_get_default_features(&features);
    features.num_channels = 1;
    features.num_image_rows = 28;
    features.num_image_cols = 28;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.0001;
    features.mean = 0;
    features.pad = 0;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 5;
    features.filter_cols = 5;
    features.output_depth = 20;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 20;
    features.bottom_layer = 0;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 2;
    features.stride = 1;
    features.ReLU = 1;
    features.batch_norm = 0;
    features.filter_rows = 5;
    features.filter_cols = 5;
    features.output_depth = 50;
    features.bottom_layer = 1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 50;
    features.bottom_layer = 2;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.1;
    features.pad = 0;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_cols = 1;
    features.filter_rows = 64;
    features.output_depth = 1;
    features.bottom_layer = 3;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.1;
    features.pad = 0;
    features.stride = 1;
    features.loss_type = LOSS_TYPE_SOFTMAX;
    features.filter_cols = 1;
    features.filter_rows = 10;
    features.output_depth = 1;
    features.bottom_layer = 4;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_cifar10(struct model_t *model, struct feeder_t *feeder)
{
    struct feature_t features;

    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 32;
    features.num_image_cols = 32;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.001;
    features.mean = 0;
    features.pad = 2;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 5;
    features.filter_cols = 5;
    features.output_depth = 32;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 32;
    features.bottom_layer = 0;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 2;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 5;
    features.filter_cols = 5;
    features.output_depth = 32;
    features.bottom_layer = 1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 32;
    features.bottom_layer = 2;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 2;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 5;
    features.filter_cols = 5;
    features.output_depth = 64;
    features.bottom_layer = 3;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 64;
    features.bottom_layer = 4;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.1;
    features.pad = 0;
    features.stride = 1;
    features.filter_cols = 1;
    features.filter_rows = 64;
    features.output_depth = 1;
    features.bottom_layer = 5;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.1;
    features.pad = 0;
    features.stride = 1;
    features.loss_type = LOSS_TYPE_SOFTMAX;
    features.filter_cols = 1;
    features.filter_rows = 10;
    features.output_depth = 1;
    features.bottom_layer = 6;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_vgga(struct model_t *model, struct feeder_t *feeder)
{
    struct feature_t features;

    /* 0 conv */
    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 224;
    features.num_image_cols = 224;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 64;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    /* 1 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 64;
    features.bottom_layer = 0;
    pcnn_model_init_layer(model, feeder, &features);

    /* 2 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 128;
    features.bottom_layer = 1;
    pcnn_model_init_layer(model, feeder, &features);

    /* 3 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 128;
    features.bottom_layer = 2;
    pcnn_model_init_layer(model, feeder, &features);

    /* 4 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 256;
    features.bottom_layer = 3;
    pcnn_model_init_layer(model, feeder, &features);

    /* 5 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 256;
    features.bottom_layer = 4;
    pcnn_model_init_layer(model, feeder, &features);

    /* 6 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.ReLU = 0;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 256;
    features.bottom_layer = 5;
    pcnn_model_init_layer(model, feeder, &features);

    /* 7 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 512;
    features.bottom_layer = 6;
    pcnn_model_init_layer(model, feeder, &features);

    /* 8 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 512;
    features.bottom_layer = 7;
    pcnn_model_init_layer(model, feeder, &features);

    /* 9 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 512;
    features.bottom_layer = 8;
    pcnn_model_init_layer(model, feeder, &features);

    /* 10 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 512;
    features.bottom_layer = 9;
    pcnn_model_init_layer(model, feeder, &features);

    /* 11 conv */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 512;
    features.bottom_layer = 10;
    pcnn_model_init_layer(model, feeder, &features);

    /* 12 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 2;
    features.filter_cols = 2;
    features.output_depth = 512;
    features.bottom_layer = 11;
    pcnn_model_init_layer(model, feeder, &features);

    /* 13 full */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.01;
    features.pad = 0;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_cols = 1;
    features.filter_rows = 4096;
    features.output_depth = 1;
    features.bottom_layer = 12;
    pcnn_model_init_layer(model, feeder, &features);

    /* 14 full */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.01;
    features.pad = 0;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_cols = 1;
    features.filter_rows = 4096;
    features.output_depth = 1;
    features.bottom_layer = 13;
    pcnn_model_init_layer(model, feeder, &features);

    /* 15 full */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.01;
    features.pad = 0;
    features.stride = 1;
    features.loss_type = LOSS_TYPE_SOFTMAX;
    features.filter_cols = 1;
    features.filter_rows = 1000;
    features.output_depth = 1;
    features.bottom_layer = 14;
    pcnn_model_init_layer(model, feeder, &features);
}

static void pcnn_arch_resblock_large(struct model_t *model, struct feeder_t *feeder, int depth0, int depth1, int depth2, int bottom_layer, int stride)
{
    int skip_from;
    struct layer_t *bottom;
    struct feature_t features;

    bottom = model->layers[bottom_layer];

    /* 1 conv (left branch) */
    if(bottom->output_depth != depth2){
        pcnn_model_get_default_features(&features);
        features.type = LAYER_TYPE_CONV;
        features.std = 0.01;
        features.mean = 0;
        features.pad = 0;
        features.stride = stride;
        features.batch_norm = 1;
        features.filter_rows = 1;
        features.filter_cols = 1;
        features.output_depth = depth2;
        features.bottom_layer = bottom->id;
        pcnn_model_init_layer(model, feeder, &features);
        skip_from = model->layers[model->num_layers-1]->id;
    }
    else{
        skip_from = bottom_layer;
    }

    /* 2 conv (right branch) */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 0;
    features.stride = stride;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.filter_rows = 1;
    features.filter_cols = 1;
    features.output_depth = depth0;
    features.bottom_layer = bottom->id;
    pcnn_model_init_layer(model, feeder, &features);

    /* 3 conv (right branch) */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = depth1;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);

    /* 4 conv (right branch) */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 0;
    features.stride = 1;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.filter_rows = 1;
    features.filter_cols = 1;
    features.output_depth = depth2;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    features.skip_from = skip_from;
    pcnn_model_init_layer(model, feeder, &features);
}

static void pcnn_arch_resblock_small(struct model_t *model, struct feeder_t *feeder, int depth0, int depth1, int bottom_layer, int stride)
{
    int skip_from;
    struct layer_t *bottom;
    struct feature_t features;

    bottom = model->layers[bottom_layer];

    /* 1 conv (left branch) */
    if(bottom->output_depth != depth1){
        pcnn_model_get_default_features(&features);
        features.type = LAYER_TYPE_CONV;
        features.std = 0.01;
        features.mean = 0;
        features.pad = 0;
        features.stride = stride;
        features.batch_norm = 1;
        features.filter_rows = 1;
        features.filter_cols = 1;
        features.output_depth = depth1;
        features.bottom_layer = bottom->id;
        pcnn_model_init_layer(model, feeder, &features);
        skip_from = model->layers[model->num_layers-1]->id;
    }
    else{
        skip_from = bottom_layer;
    }

    /* 2 conv (right branch) */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = stride;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = depth0;
    features.bottom_layer = bottom->id;
    pcnn_model_init_layer(model, feeder, &features);

    /* 4 conv (right branch) */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = depth1;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    features.skip_from = skip_from;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_resnet20(struct model_t *model, struct feeder_t *feeder)
{
    struct feature_t features;

    /* 0 conv */
    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 32;
    features.num_image_cols = 32;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.output_depth = 16;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_arch_resblock_small(model, feeder, 16, 16, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_small(model, feeder, 16, 16, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_small(model, feeder, 16, 16, model->layers[model->num_layers-1]->id, 1);

    pcnn_arch_resblock_small(model, feeder, 32, 32, model->layers[model->num_layers-1]->id, 2);
    pcnn_arch_resblock_small(model, feeder, 32, 32, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_small(model, feeder, 32, 32, model->layers[model->num_layers-1]->id, 1);

    pcnn_arch_resblock_small(model, feeder, 64, 64, model->layers[model->num_layers-1]->id, 2);
    pcnn_arch_resblock_small(model, feeder, 64, 64, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_small(model, feeder, 64, 64, model->layers[model->num_layers-1]->id, 1);

    /* final pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 1; /* global average pooling */
    features.pad = 0;
    features.stride = 1;
    features.filter_rows = model->layers[model->num_layers-1]->output_rows;
    features.filter_cols = model->layers[model->num_layers-1]->output_cols;
    features.output_depth = model->layers[model->num_layers-1]->output_depth;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);

    /* final full */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.01;
    features.pad = 0;
    features.stride = 1;
    features.filter_cols = 1;
    features.filter_rows = 10;
    features.output_depth = 1;
    features.loss_type = LOSS_TYPE_SOFTMAX;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_resnet50(struct model_t *model, struct feeder_t *feeder)
{
    struct feature_t features;

    /* 0 conv */
    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 224;
    features.num_image_cols = 224;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 3;
    features.stride = 2;
    features.filter_rows = 7;
    features.filter_cols = 7;
    features.ReLU = 1;
    features.batch_norm = 1;
    features.output_depth = 64;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    /* 1 pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 0; /* max-pooling */
    features.pad = 0;
    features.stride = 2;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 64;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_arch_resblock_large(model, feeder, 64, 64, 256, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 64, 64, 256, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 64, 64, 256, model->layers[model->num_layers-1]->id, 1);

    pcnn_arch_resblock_large(model, feeder, 128, 128, 512, model->layers[model->num_layers-1]->id, 2);
    pcnn_arch_resblock_large(model, feeder, 128, 128, 512, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 128, 128, 512, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 128, 128, 512, model->layers[model->num_layers-1]->id, 1);

    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 2);
    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 256, 256, 1024, model->layers[model->num_layers-1]->id, 1);

    pcnn_arch_resblock_large(model, feeder, 512, 512, 2048, model->layers[model->num_layers-1]->id, 2);
    pcnn_arch_resblock_large(model, feeder, 512, 512, 2048, model->layers[model->num_layers-1]->id, 1);
    pcnn_arch_resblock_large(model, feeder, 512, 512, 2048, model->layers[model->num_layers-1]->id, 1);

    /* final pool */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_POOL;
    features.sub_type = 1; /* global average pooling */
    features.pad = 0;
    features.stride = 1;
    features.filter_rows = model->layers[model->num_layers-1]->output_rows;
    features.filter_cols = model->layers[model->num_layers-1]->output_cols;
    features.output_depth = model->layers[model->num_layers-1]->output_depth;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);

    /* final full */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_FULL;
    features.std = 0.01;
    features.pad = 0;
    features.stride = 1;
    features.filter_cols = 1;
    features.filter_rows = 1000;
    features.output_depth = 1;
    features.loss_type = LOSS_TYPE_SOFTMAX;
    features.bottom_layer = model->layers[model->num_layers-1]->id;
    pcnn_model_init_layer(model, feeder, &features);
}

static void pcnn_arch_edsr_resblock(struct model_t *model, struct feeder_t *feeder, int num_kernels, int bottom, int batch_norm, float scale)
{
    struct feature_t features;

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = bottom;
    features.batch_norm = batch_norm;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = bottom+1;
    features.skip_from = bottom;
    features.res_scale = scale;
    features.batch_norm = batch_norm;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_edsr(struct model_t *model, struct feeder_t *feeder)
{
    int i;
    int num_kernels = 256;
    int num_res_blocks = 32;
    struct feature_t features;

    /* HEAD */
    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 48;
    features.num_image_cols = 48;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    /* BODY */
    for(i=0; i<num_res_blocks; i++)
        pcnn_arch_edsr_resblock(model, feeder, num_kernels, model->num_layers-1, 0, 0.1f);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 1;
    features.stride = 1;
    features.filter_cols = 3;
    features.filter_rows = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = model->num_layers-1;
    features.skip_from = 0;
    pcnn_model_init_layer(model, feeder, &features);

    /* TAIL */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 1;
    features.stride = 1;
    features.filter_cols = 3;
    features.filter_rows = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = model->num_layers-1;
    pcnn_model_init_layer(model, feeder, &features);

    num_kernels = 3 * model->upsample_ratio * model->upsample_ratio;

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 1;
    features.stride = 1;
    features.filter_cols = 3;
    features.filter_rows = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = model->num_layers-1;
    pcnn_model_init_layer(model, feeder, &features);

    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_UPSAMPLE;
    features.std = 0.01;
    features.loss_type = LOSS_TYPE_MAE;
    features.bottom_layer = model->num_layers-1;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_drrn(struct model_t *model, struct feeder_t *feeder)
{
    int i, j;
    int U, B;
    int bottom;
    int num_kernels = 128;
    struct feature_t features;

    B = 1;
    U = 25;

    /* HEAD */
    pcnn_model_get_default_features(&features);
    features.num_channels = 1;
    features.num_image_rows = 32;
    features.num_image_cols = 32;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    /* BODY */
    for(i=0; i<B; i++){
        bottom = model->num_layers-1;
        for(j=0; j<U; j++){
            pcnn_arch_edsr_resblock(model, feeder, num_kernels, bottom, 1, 1.f);
        }
    }

    /* TAIL */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.pad = 1;
    features.stride = 1;
    features.loss_type = LOSS_TYPE_MAE;
    features.filter_cols = 3;
    features.filter_rows = 3;
    features.output_depth = 1;
    features.skip_from = 0;
    features.bottom_layer = model->num_layers-1;
    pcnn_model_init_layer(model, feeder, &features);
}

void pcnn_arch_config_vdsr(struct model_t *model, struct feeder_t *feeder)
{
    int i;
    int num_kernels = 64;
    int num_body_layers = 20;
    struct feature_t features;

    /* HEAD */
    pcnn_model_get_default_features(&features);
    features.num_channels = 3;
    features.num_image_rows = 32;
    features.num_image_cols = 32;
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 0;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = num_kernels;
    features.bottom_layer = -1;
    pcnn_model_init_layer(model, feeder, &features);

    /* BODY */
    for (i=0; i<num_body_layers; i++) {
        pcnn_model_get_default_features(&features);
        features.type = LAYER_TYPE_CONV;
        features.std = 0.01;
        features.mean = 0;
        features.pad = 1;
        features.stride = 1;
        features.ReLU = 1;
        features.filter_rows = 3;
        features.filter_cols = 3;
        features.output_depth = num_kernels;
        features.bottom_layer = model->num_layers - 1;
        pcnn_model_init_layer(model, feeder, &features);
    }

    /* TAIL */
    pcnn_model_get_default_features(&features);
    features.type = LAYER_TYPE_CONV;
    features.std = 0.01;
    features.mean = 0;
    features.pad = 1;
    features.stride = 1;
    features.ReLU = 0;
    features.filter_rows = 3;
    features.filter_cols = 3;
    features.output_depth = 3;
    features.bottom_layer = model->num_layers - 1;
    features.loss_type = LOSS_TYPE_MAE;
    features.skip_from = -1; // input is directly added to the activations
    pcnn_model_init_layer(model, feeder, &features);
}
