/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_MKL 
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include "def.h"
#include "config.h"
#include "model.h"
#include "feeder.h"
#include "comm.h"
#include "util.h"
#include "record.h"
#include "batch_norm.h"

/* This is a model module which organizes all the model parameters
 * and the intermediate variables such as activations, erros, and gradients. */
static void pcnn_model_calc_comm_offsets(struct layer_t *layer,
                                         struct model_t *model,
                                         struct comm_queue_t *queue)
{
    int i;
    int size;
    int remainder;

    if(layer->type != LAYER_TYPE_CONV && layer->type != LAYER_TYPE_FULL)
        return;

    /* weight parameters communication data offsets */
    size = layer->filter_size / queue->nproc;
    remainder = layer->filter_size % queue->nproc;
    layer->local_weight_count = size + ((queue->rank < remainder) ? 1 : 0);
    layer->local_weight_off = (size * queue->rank) +
                              ((queue->rank < remainder) ? queue->rank : remainder);
    if(remainder != 0){
        layer->aligned_weight = 0;
        layer->sdispls_weight = (int *)calloc(queue->nproc, sizeof(int));
        layer->rdispls_weight = (int *)calloc(queue->nproc, sizeof(int));
        layer->scounts_weight = (int *)calloc(queue->nproc, sizeof(int));
        layer->rcounts_weight = (int *)calloc(queue->nproc, sizeof(int));

        /* send/receive counts */
        for(i=0; i<queue->nproc; i++){
            layer->scounts_weight[i] = ((i < remainder) ? size + 1 : size);
            layer->rcounts_weight[i] = layer->local_weight_count;
        }

        /* send/receive displacement */
        layer->sdispls_weight[0] = 0;
        layer->rdispls_weight[0] = 0;
        for(i=1; i<queue->nproc; i++){
            layer->sdispls_weight[i] = layer->sdispls_weight[i-1] + layer->scounts_weight[i-1];
            layer->rdispls_weight[i] = layer->rdispls_weight[i-1] + layer->rcounts_weight[i-1];
        }
    }
    else{
        layer->aligned_weight = 1; 
    }

    /* weight + bias parameters communication data offsets */
    size = layer->num_gradients / queue->nproc;
    remainder = layer->num_gradients % queue->nproc;
    layer->num_local_gradients = size + ((queue->rank < remainder) ? 1 : 0);

    if(remainder > 0){
        layer->aligned_gradients = 0;
        layer->sdispls_gradients = (int *)calloc(queue->nproc, sizeof(int));
        layer->rdispls_gradients = (int *)calloc(queue->nproc, sizeof(int));
        layer->scounts_gradients = (int *)calloc(queue->nproc, sizeof(int));
        layer->rcounts_gradients = (int *)calloc(queue->nproc, sizeof(int));

        /* send/receive counts */
        for(i=0; i<queue->nproc; i++){
            layer->scounts_gradients[i] = ((i < remainder) ? size + 1 : size);
            layer->rcounts_gradients[i] = layer->num_local_gradients;
        }

        /* send/receive displacement */
        layer->sdispls_gradients[0] = 0;
        layer->rdispls_gradients[0] = 0;
        for(i=1; i<queue->nproc; i++){
            layer->sdispls_gradients[i] = layer->sdispls_gradients[i-1] + layer->scounts_gradients[i-1];
            layer->rdispls_gradients[i] = layer->rdispls_gradients[i-1] + layer->rcounts_gradients[i-1];
        }
    }
    else{
        layer->aligned_gradients = 1; 
    }
}

static void pcnn_model_init_param_values(struct param_t *param, struct model_t *model, struct comm_queue_t *queue)
{
    int i=0, j=0;
    int fan_in, fan_out;
    struct layer_t *layer;

    if(queue->global_rank == 0){
        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];

            /* Calculate the standard deviation value depending on the 
            * weight initialization method. */
            if(model->param_init_method == PARAM_INIT_TYPE_MSRA){
                fan_out = layer->output_channels *
                          layer->filter_depth *
                          layer->filter_rows *
                          layer->filter_cols;
                layer->std = sqrtf(2.0f / (float)fan_out);
            }
            else if(model->param_init_method == PARAM_INIT_TYPE_GLOROT){
                fan_in = layer->input_channels *
                         layer->filter_depth *
                         layer->filter_rows *
                         layer->filter_cols;
                fan_out = layer->output_channels *
                          layer->filter_depth *
                          layer->filter_rows *
                          layer->filter_cols;
                layer->std = sqrtf(2.0f / (float)(fan_in + fan_out));
            }

            /* Initialize the parameters. */
            if(layer->type == LAYER_TYPE_CONV){
                pcnn_util_gaussian(layer->id, layer->filter_size, layer->mean, layer->std, layer->weight);
                for(j=0; j<layer->output_channels; j++){
                    layer->bias[j] = 0;
                }
            }
            else if(layer->type == LAYER_TYPE_FULL){
                pcnn_util_gaussian(layer->id, layer->filter_size, layer->mean, layer->std, layer->weight);
                for(j=0; j<layer->num_neurons; j++){
                    layer->bias[j] = 0;
                }
            }
        }
    }
    MPI_Bcast(param->params, param->total_size, MPI_FLOAT, 0, queue->world);
}

struct model_t *pcnn_model_init(int num_train_images, int num_epochs, int mode, struct comm_queue_t *queue)
{
    struct model_t *model;

    model = (struct model_t *)malloc(sizeof(struct model_t));
    memset(model, 0, sizeof(struct model_t));

    model->num_epochs = num_epochs;
    model->mode = mode;

    /* Settings from config.h */
    model->num_layers = 0;
    model->test_per_epoch = TEST_PER_EPOCH;
    model->learning_rate = LEARNING_RATE;
    model->decay_factor = LEARNING_RATE_DECAY_FACTOR;
    model->decay_steps = LEARNING_RATE_DECAY_STEPS;
    model->weight_decay = WEIGHT_DECAY;
    model->comm_pattern = COMM_PATTERN;
    model->overlap = OVERLAP;
    model->task_type = TASK_TYPE;
    model->param_init_method = PARAM_INIT_METHOD;
    model->optimizer = OPTIMIZER;
    model->momentum = MOMENTUM;
    model->beta1 = ADAM_BETA1;
    model->beta2 = ADAM_BETA2;
    model->epsilon = ADAM_EPSILON;
    model->upsample_ratio = UPSAMPLE_RATIO;
    model->eps = EPS_FACTOR;
    model->moving_average_fraction = MOVING_AVERAGE_FRACTION;
    model->checkpoint_interval = CHECKPOINT_INTERVAL;
    model->checkpoint_path = (char *)malloc(sizeof(char) * (strlen(CHECKPOINT_PATH) + 1));
    sprintf(model->checkpoint_path, "%s", (CHECKPOINT_PATH));
    model->param_size = 0;
    model->intermediate_size = 0;

    model->layers = (struct layer_t **) calloc(MAX_NUM_LAYERS, sizeof(struct layer_t *));

    if(queue->group_id == 0 && queue->rank == 0){
        printf("---------------------------------------------------------\n");
        printf("%-40s: %d\n", "Number of epochs", model->num_epochs);
        printf("%-40s: %d\n", "Number of processes", queue->nproc * queue->num_groups);
        printf("---------------------------------------------------------\n");
        if(model->optimizer == OPTIMIZER_SGD){
            printf("%-40s: %s\n", "Optimizer" , "SGD");
            printf("%-40s: %f\n", "Momentum" , model->momentum);
        }
        else if(model->optimizer == OPTIMIZER_ADAM){
            printf("%-40s: %s\n", "Optimizer" , "Adam");
            printf("%-40s: %f\n", "beta1" , model->beta1);
            printf("%-40s: %f\n", "beta2" , model->beta2);
            printf("%-40s: %f\n", "epsilon" , model->epsilon);
        }
        printf("%-40s: %f\n", "Learning rate" , model->learning_rate);
        printf("%-40s: %f\n", "Learning rate decay factor" , model->decay_factor);
        printf("%-40s: %d\n", "Learning rate decay steps" , model->decay_steps);
    }
    return model;
}

void pcnn_model_destroy(struct model_t *model)
{
    int i;
    struct layer_t *layer;

    if(model != NULL){
        if(model->num_layers > 0){
            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->a != NULL)
                    free(layer->a);
                if(layer->e != NULL)
                    free(layer->e);
                if(layer->poolmap != NULL)
                    free(layer->poolmap);
                if(layer->rep_a != NULL)
                    free(layer->rep_a);
                if(layer->rep_e != NULL)
                    free(layer->rep_e);
                if(layer->recv_a != NULL)
                    free(layer->recv_a);
                if(layer->recv_e != NULL)
                    free(layer->recv_e);
                if(layer->sdispls_weight != NULL)
                    free(layer->sdispls_weight);
                if(layer->rdispls_weight != NULL)
                    free(layer->rdispls_weight);
                if(layer->scounts_weight != NULL)
                    free(layer->scounts_weight);
                if(layer->rcounts_weight != NULL)
                    free(layer->rcounts_weight);
                if(layer->sdispls_gradients != NULL)
                    free(layer->sdispls_gradients);
                if(layer->rdispls_gradients != NULL)
                    free(layer->rdispls_gradients);
                if(layer->scounts_gradients != NULL)
                    free(layer->scounts_gradients);
                if(layer->rcounts_gradients != NULL)
                    free(layer->rcounts_gradients);
                if(layer->batch_norm == 1){
                    if(layer->a_norm != NULL)
                        free(layer->a_norm);
                    if(layer->sqrt_var != NULL)
                        free(layer->sqrt_var);
                }
            }
            free(model->layers);
            free(model->checkpoint_path);
        }
        free(model);
        model = NULL;
    }
}

void pcnn_model_init_layer(struct model_t *model, struct feeder_t *feeder, struct feature_t *features)
{
    unsigned long mem_size=0;
    struct layer_t *layer=NULL;
    struct layer_t *bottom=NULL;

    layer = (struct layer_t *)malloc(sizeof(struct layer_t));
    bottom = features->bottom_layer >= 0 ? model->layers[features->bottom_layer] : NULL;

    layer->id = model->num_layers;
    layer->type = features->type;
    layer->sub_type = features->sub_type;
    layer->ReLU = features->ReLU;
    layer->loss_type = features->loss_type;
    layer->batch_norm = features->batch_norm;
    layer->pad_depth = features->pad_depth;
    layer->pad_rows = features->pad_rows;
    layer->pad_cols = features->pad_cols;
    layer->stride_depth = features->stride_depth;
    layer->stride_rows = features->stride_rows;
    layer->stride_cols = features->stride_cols;
    layer->mean = features->mean;
    layer->std = features->std;
    layer->bottom_layer = features->bottom_layer;
    layer->skip_from = features->skip_from;
    layer->res_scale = features->res_scale;
    layer->output_channels = features->output_channels;
    layer->filter_depth = features->filter_depth;
    layer->filter_rows = features->filter_rows;
    layer->filter_cols = features->filter_cols;
    layer->bn_scale_factor = 0.0f;

    if(model->num_layers == 0){
        layer->input_channels = features->num_channels;
        layer->input_depth = features->image_depth;
        layer->input_rows = features->image_rows;
        layer->input_cols = features->image_cols;
    }
    else{
        layer->input_channels = bottom->output_channels;
        layer->input_depth = bottom->output_depth;
        layer->input_rows = bottom->output_rows;
        layer->input_cols = bottom->output_cols;
    }

    /* Calculate output spatial dimensions. */
    if(features->type == LAYER_TYPE_CONV){
        layer->output_depth = ((layer->input_depth + 2*features->pad_depth - features->filter_depth)/features->stride_depth) + 1;
        layer->output_rows = ((layer->input_rows + 2*features->pad_rows - features->filter_rows)/features->stride_rows) + 1;
        layer->output_cols = ((layer->input_cols + 2*features->pad_cols - features->filter_cols)/features->stride_cols) + 1;
    }
    else if(features->type == LAYER_TYPE_POOL){
        layer->output_depth = ceilf((float)(layer->input_depth - features->filter_depth)/features->stride_depth) + 1;
        layer->output_rows = ceilf((float)(layer->input_rows - features->filter_rows)/features->stride_rows) + 1;
        layer->output_cols = ceilf((float)(layer->input_cols - features->filter_cols)/features->stride_cols) + 1;
    }
    else if(features->type == LAYER_TYPE_FULL){
        layer->output_depth = 1;
        layer->output_cols = 1;
        layer->output_rows = features->filter_rows; /* In a fully-connected layer, filter_rows is the number of neurons. */
    }
    else if(features->type == LAYER_TYPE_UPSAMPLE){
        layer->output_depth = 1;
        layer->output_rows = layer->input_rows * model->upsample_ratio;
        layer->output_cols = layer->input_cols * model->upsample_ratio;
        layer->output_channels = layer->input_channels / (model->upsample_ratio * model->upsample_ratio);
    }

    layer->num_neurons = layer->output_channels * layer->output_depth * layer->output_rows * layer->output_cols;
    layer->num_prev_neurons = layer->input_channels * layer->input_depth * layer->input_rows * layer->input_cols;

    /* weight offset and length calculation */
    if(layer->type == LAYER_TYPE_CONV){
        layer->filter_size = layer->output_channels *
                             layer->input_channels *
                             layer->filter_depth *
                             layer->filter_rows *
                             layer->filter_cols;
        layer->bias_size = layer->output_channels;
        layer->num_gradients = layer->filter_size + layer->bias_size;
    }
    else if(layer->type == LAYER_TYPE_FULL){
        layer->filter_size = layer->num_neurons * layer->num_prev_neurons;
        layer->bias_size = layer->num_neurons;
        layer->num_gradients = layer->filter_size + layer->bias_size;
    }
    else{
        layer->filter_size = 0;
        layer->bias_size = 0;
        layer->num_gradients = 0;
    }

    /* Allocate memory spaces. */
    layer->a = NULL;
    layer->e = NULL;
    layer->poolmap = NULL;
    layer->rep_a = NULL;
    layer->rep_e = NULL;
    layer->recv_a = NULL;
    layer->recv_e = NULL;
    layer->a_norm = NULL;
    layer->sqrt_var = NULL;
    layer->global_mean = NULL;
    layer->global_variance = NULL;
    layer->gamma = NULL;
    layer->beta = NULL;
    layer->sdispls_weight = NULL;
    layer->rdispls_weight = NULL;
    layer->scounts_weight = NULL;
    layer->rcounts_weight = NULL;
    layer->sdispls_gradients = NULL;
    layer->rdispls_gradients = NULL;
    layer->scounts_gradients = NULL;
    layer->rcounts_gradients = NULL;

    if(layer->type == LAYER_TYPE_FULL){
        layer->a = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        layer->e = (float *)calloc(layer->num_neurons * feeder->batch_size, sizeof(float));
        layer->rep_e = (float *)calloc(layer->num_neurons * feeder->batch_size, sizeof(float));
        layer->recv_e = (float *)calloc(layer->num_neurons * feeder->batch_size, sizeof(float));

        if(bottom != NULL){
            bottom->rep_a = (float *)calloc(feeder->local_batch_size * bottom->num_neurons, sizeof(float));
            bottom->recv_a = (float *)calloc(feeder->local_batch_size * bottom->num_neurons, sizeof(float));
        }
    }
    else if(layer->type == LAYER_TYPE_POOL){
        layer->a = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        layer->e = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        layer->poolmap = (int *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(int));
    }
    else if(layer->type == LAYER_TYPE_CONV){
        layer->a = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        layer->e = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        if(layer->batch_norm){
            layer->a_norm = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
            layer->sqrt_var = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        }
    }
    else if(layer->type == LAYER_TYPE_UPSAMPLE){
        layer->a = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
        layer->e = (float *)calloc(feeder->local_batch_size * layer->num_neurons, sizeof(float));
    }

    /* statistics */
    if(layer->type == LAYER_TYPE_FULL){
        mem_size += feeder->local_batch_size * layer->num_neurons; // a
        mem_size += feeder->local_batch_size * layer->num_neurons; // rep_a
        mem_size += feeder->local_batch_size * layer->num_neurons; // recv_a
        mem_size += feeder->batch_size * layer->num_neurons; // e 
        mem_size += feeder->batch_size * layer->num_neurons; // rep_e
        mem_size += feeder->batch_size * layer->num_neurons; // recv_e
    }
    else if(layer->type == LAYER_TYPE_POOL){
        mem_size += feeder->local_batch_size * layer->num_neurons; // a
        mem_size += feeder->local_batch_size * layer->num_neurons; // rep_a
        mem_size += feeder->local_batch_size * layer->num_neurons; // recv_a
        mem_size += feeder->local_batch_size * layer->num_neurons; // e
        mem_size += feeder->local_batch_size * layer->num_neurons; // poolmap
    }
    else if(layer->type == LAYER_TYPE_CONV){
        mem_size += feeder->local_batch_size * layer->num_neurons; // a
        mem_size += feeder->local_batch_size * layer->num_neurons; // e
        if(layer->batch_norm){
            mem_size += feeder->local_batch_size * layer->num_neurons; // a_norm
            mem_size += feeder->local_batch_size * layer->num_neurons; // sqrt_var
            mem_size += layer->output_channels; // gamma
            mem_size += layer->output_channels; // beta
        }
    }
    else if(layer->type == LAYER_TYPE_UPSAMPLE){
        mem_size += feeder->local_batch_size * layer->num_neurons; // a
        mem_size += feeder->local_batch_size * layer->num_neurons; // e
    }
    mem_size *= sizeof(float);

    model->layers[model->num_layers++] = layer;
    model->intermediate_size += mem_size;
}

struct param_t *pcnn_model_init_param(struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    size_t i=0, j=0;
    size_t rows=0;
    size_t cols=0;
    size_t count=0;
    size_t offset=0;
    size_t accum_size=0;
    size_t max_size=0;
    size_t this_layer_max=0;
    size_t pool2full_size=0;
    size_t conv_weight_size=0;
    size_t conv_bias_size=0;
    size_t conv_total_size=0;
    size_t full_weight_size=0;
    size_t full_bias_size=0;
    size_t full_total_size=0;
    size_t total_size=0;
    size_t bn_param_size=0;
    struct param_t *param=NULL;
    struct layer_t *layer=NULL;

    param = (struct param_t *)calloc(1, sizeof(struct param_t));

    /* Calculate the size of parameters. */
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];

        if(layer->type == LAYER_TYPE_CONV){
            conv_weight_size += (layer->output_channels *
                                 layer->input_channels *
                                 layer->filter_depth *
                                 layer->filter_rows *
                                 layer->filter_cols);
            conv_bias_size += layer->output_channels; /* bias parameters */
        }
        else if(layer->type == LAYER_TYPE_FULL){
            full_weight_size += layer->num_neurons * layer->num_prev_neurons;
            full_bias_size += layer->num_neurons;
        }
    }
    conv_total_size = conv_weight_size + conv_bias_size;
    full_total_size = full_weight_size + full_bias_size;
    total_size = conv_total_size + full_total_size;

    /* Count the number of batch normalization tunable parameters. */
    bn_param_size = 0;
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];
        if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
            bn_param_size += layer->output_channels; // gamma
            bn_param_size += layer->output_channels; // beta
        }
    }

    param->conv_weight_size = conv_weight_size;
    param->conv_bias_size = conv_bias_size;
    param->conv_total_size = conv_total_size;
    param->full_weight_size = full_weight_size;
    param->full_bias_size = full_bias_size;
    param->full_total_size = full_total_size;
    param->total_size = total_size;
    param->bn_param_size = bn_param_size;
    param->num_updates = 0;
    param->num_trained_epochs = 0;
    param->beta1_decay = 1.f;
    param->beta2_decay = 1.f;

    /* Allocate memory spaces for parameters and gradients. 
     * Here, we need at least four sets of full-sized memory space:
     * a set of parameters: param->params
     * a sending buffer for gradients: param->gradients
     * a receiving buffer for gradients: param->gradient_sums
     * a set of previous gradient sums: param->prev_gradient_sums
     * Note that the previous gradient sums are used for momentum. */
    param->params = (float *)calloc(total_size, sizeof(float));
    param->gradients = (float *)calloc(total_size, sizeof(float));
    param->gradient_sums = (float *)calloc(total_size, sizeof(float));
    if(model->optimizer == OPTIMIZER_SGD){
        param->prev_gradient_sums = (float *)calloc(total_size, sizeof(float));
        param->m_gradient_sums = NULL;
        param->v_gradient_sums = NULL;
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        param->prev_gradient_sums = NULL;
        param->m_gradient_sums = (float *)calloc(total_size, sizeof(float));
        param->v_gradient_sums = (float *)calloc(total_size, sizeof(float));
    }

    /* Memory allocation for weight, bias, and gradients.
     * Given a single large memory space, weights and biases are allocated for each layer.
     * Later, when averaging the gradients, we merge the weight and bias gradients and 
     * perform multi-step communications on it at once.
     * So, L2 regularization is always applied to weights and biases together. 
     */
    offset = 0;
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];

        if(layer->type == LAYER_TYPE_CONV){
            layer->weight = &param->params[offset];
            layer->local_sumws = &param->gradients[offset];
            layer->global_sumws = &param->gradient_sums[offset];
            if(model->optimizer == OPTIMIZER_SGD){
                layer->prev_sumws = &param->prev_gradient_sums[offset];
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                layer->m_sumws = &param->m_gradient_sums[offset];
                layer->v_sumws = &param->v_gradient_sums[offset];
            }
            offset += (layer->output_channels *
                       layer->input_channels *
                       layer->filter_depth *
                       layer->filter_rows *
                       layer->filter_cols);

            layer->bias = &param->params[offset];
            layer->local_sumbs = &param->gradients[offset];
            layer->global_sumbs = &param->gradient_sums[offset];
            if(model->optimizer == OPTIMIZER_SGD){
                layer->prev_sumbs = &param->prev_gradient_sums[offset];
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                layer->m_sumbs = &param->m_gradient_sums[offset];
                layer->v_sumbs = &param->v_gradient_sums[offset];
            }
            offset += layer->output_channels;
        }
        else if(layer->type == LAYER_TYPE_FULL){
            layer->weight = &param->params[offset];
            layer->local_sumws = &param->gradients[offset];
            layer->global_sumws = &param->gradient_sums[offset];
            if(model->optimizer == OPTIMIZER_SGD){
                layer->prev_sumws = &param->prev_gradient_sums[offset];
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                layer->m_sumws = &param->m_gradient_sums[offset];
                layer->v_sumws = &param->v_gradient_sums[offset];
            }
            offset += layer->num_neurons * layer->num_prev_neurons;

            layer->bias = &param->params[offset];
            layer->local_sumbs = &param->gradients[offset];
            layer->global_sumbs = &param->gradient_sums[offset];
            if(model->optimizer == OPTIMIZER_SGD){
                layer->prev_sumbs = &param->prev_gradient_sums[offset];
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                layer->m_sumbs = &param->m_gradient_sums[offset];
                layer->v_sumbs = &param->v_gradient_sums[offset];
            }
            offset += layer->num_neurons;
        }
    }

    /* Allocate a memory space for batch normalization parameters. */
    param->bn_params = NULL;
    param->bn_gradients = NULL;
    param->bn_gradient_sums = NULL;
    param->bn_prev_gradients = NULL;
    param->bn_m_gradients = NULL;
    param->bn_v_gradients = NULL;
    if(bn_param_size > 0){
        param->bn_params = (float *)calloc(bn_param_size, sizeof(float));
        param->bn_gradients = (float *)calloc(bn_param_size, sizeof(float));
        if(queue->nproc > 1)
            param->bn_gradient_sums = (float *)calloc(bn_param_size, sizeof(float));

        if(model->optimizer == OPTIMIZER_SGD){
            param->bn_prev_gradients = (float *)calloc(bn_param_size, sizeof(float));
        }
        else if(model->optimizer == OPTIMIZER_ADAM){
            param->bn_m_gradients = (float *)calloc(bn_param_size, sizeof(float));
            param->bn_v_gradients = (float *)calloc(bn_param_size, sizeof(float));
        }

        /* Calculate offsets for each batch normalization layer and assign
         * a sub-region of the memory space. */
        offset = 0;
        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];
            if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                /* gamma (scaling parameters) */
                layer->gamma = &param->bn_params[offset];
                offset += layer->output_channels;
                for(j=0; j<layer->output_channels; j++)
                    layer->gamma[j] = 1.0f;

                /* beta (shift parameters) */
                layer->beta = &param->bn_params[offset];
                offset += layer->output_channels;
            }
        }

        /* Assign a sub-region of memory space for the batch normalization
         * gamma gradients at each layer. */
        offset = 0;
        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];
            if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                layer->local_dgamma = &param->bn_gradients[offset];
                offset += layer->output_channels;
            }
        }

        /* Assign a sub-region of memory space for the batch normalization
         * beta gradients at each layer. */
        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];
            if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                layer->local_dbeta = &param->bn_gradients[offset];
                offset += layer->output_channels;
            }
        }

        /* momentum parameters for batch normalization */
        if(model->optimizer == OPTIMIZER_SGD){
            offset = 0;
            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->prev_dgamma = &param->bn_prev_gradients[offset];
                    offset += layer->output_channels;
                }
            }

            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->prev_dbeta = &param->bn_prev_gradients[offset];
                    offset += layer->output_channels;
                }
            }
        }
        else if(model->optimizer == OPTIMIZER_ADAM){
            offset = 0;
            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->m_dgamma = &param->bn_m_gradients[offset];
                    layer->v_dgamma = &param->bn_v_gradients[offset];
                    offset += layer->output_channels;
                }
            }

            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->m_dbeta = &param->bn_m_gradients[offset];
                    layer->v_dbeta = &param->bn_v_gradients[offset];
                    offset += layer->output_channels;
                }
            }
        }
        else{
            printf("[%s][%s][%d] Invalid optimizer setting!\n", __FILE__, __FUNCTION__, __LINE__);
            return NULL;
        }

        if(queue->nproc > 1){
            /* Assign a sub-region of memory space for the batch normalization
             * gamma gradient sums at each layer. */
            offset = 0;
            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->global_dgamma = &param->bn_gradient_sums[offset];
                    offset += layer->output_channels;
                }
            }

            /* Assign a sub-region of memory space for the batch normalization
             * beta gradient sums at each layer. */
            for(i=0; i<model->num_layers; i++){
                layer = model->layers[i];
                if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                    layer->global_dbeta = &param->bn_gradient_sums[offset];
                    offset += layer->output_channels;
                }
            }
        }
    }

    /* Allocate a column buffer for im2col and col2im operations. */
    max_size = 0;
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];

        if(layer->type == LAYER_TYPE_CONV){
            rows = feeder->local_batch_size *
                   layer->output_depth *
                   layer->output_rows *
                   layer->output_cols;
            cols = layer->input_channels *
                   layer->filter_depth *
                   layer->filter_rows *
                   layer->filter_cols;

            this_layer_max = rows * cols;
            if(max_size < this_layer_max)
                max_size = this_layer_max;
        }
    }
    param->col = (float *)calloc(max_size, sizeof(float));

    /* Find the first fully-connected layer and keep the layer ID. */
    param->first_full_id = -1;
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];
        if(layer->type == LAYER_TYPE_FULL){
            param->first_full_id = layer->id;
            break;
        }
    }

    /* Allocate a pool2full buffer for data layout transformation between
     * the last pooling layer and the first fully-connected layer. */
    if(param->first_full_id > -1){
        layer = model->layers[param->first_full_id-1];
        pool2full_size = layer->output_channels *
                         layer->output_depth *
                         layer->output_rows *
                         layer->output_cols *
                         feeder->local_batch_size;
        param->pool2full = (float *)calloc(pool2full_size, sizeof(float));
    }

    /* Initialize the parameter values. */
    pcnn_model_init_param_values(param, model, queue);

    /* Multiplier for batch normalization.
    * This array will be reused across all the layers.
    * So, find the largest layer and assign a memory space for that. */
    param->multiplier = NULL;
    param->sums = NULL;

    max_size = 0;
    
    for(i=0; i<model->num_layers; i++){
        layer =model->layers[i];
        if(layer->type == LAYER_TYPE_CONV){
            if(max_size < layer->output_rows * layer->output_cols * layer->output_depth * layer->output_channels * feeder->local_batch_size)
                max_size = layer->output_rows * layer->output_cols * layer->output_depth * layer->output_channels * feeder->local_batch_size;
        }
    }

    if(max_size > 0){
        param->multiplier = (float *)malloc(sizeof(float) * max_size);
#pragma omp parallel for
        for(i=0; i<max_size; i++)
            param->multiplier[i] = 1.0f;
    }

    /* Batch normalization-related memory space. */
    max_size = 0;
    count = 0;
    param->bn_num_layers = 0;
    for(i=0; i<model->num_layers; i++){
        layer =model->layers[i];
        if(layer->type == LAYER_TYPE_CONV && layer->batch_norm){
            param->bn_num_layers++;
            count += layer->output_channels;
            if(max_size < layer->output_channels)
                max_size = layer->output_channels;
        }
    }

    if(max_size > 0){
        max_size *= feeder->local_batch_size;
        param->sums = (float *)malloc(sizeof(float) * max_size);
    }

    param->bn_global_statistics_size = count * 2; // One for global mean and another for global variance.
    if(param->bn_global_statistics_size > 0)
        param->bn_global_statistics = (float *)calloc(param->bn_global_statistics_size, sizeof(float));

    /* Assign a space for global means first. */
    offset = 0;
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];
        if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
            layer->global_mean = &param->bn_global_statistics[offset];
            offset += layer->output_channels;
        }
    }

    /* Assign another space for global variance. */
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];
        if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
            layer->global_variance = &param->bn_global_statistics[offset];
            offset += layer->output_channels;
        }
    }

    /* statistics */
    model->param_size = total_size * sizeof(float);
    model->intermediate_size += (3*total_size + max_size + pool2full_size)*sizeof(float);

    return param;
}

void pcnn_model_free_param(struct model_t *model, struct param_t *param)
{
    if(model == NULL){
        printf("[%s][%d] model is NULL\n", __FUNCTION__,__LINE__);
        return;
    }

    if(param != NULL){
        if(param->prev_gradient_sums != NULL)
            free(param->prev_gradient_sums);
        if(param->m_gradient_sums != NULL)
            free(param->m_gradient_sums);
        if(param->v_gradient_sums != NULL)
            free(param->v_gradient_sums);
        if(param->local_conv_grads != NULL)
            free(param->local_conv_grads);
        if(param->global_conv_grads != NULL)
            free(param->global_conv_grads);
        if(param->prev_conv_grads != NULL)
            free(param->prev_conv_grads);
        if(param->local_full_grads != NULL)
            free(param->local_full_grads);
        if(param->global_full_grads != NULL)
            free(param->global_full_grads);
        if(param->prev_full_grads != NULL)
            free(param->prev_full_grads);
        if(param->params != NULL)
            free(param->params);
        if(param->gradients != NULL)
            free(param->gradients);
        if(param->bn_global_statistics != NULL)
            free(param->bn_global_statistics);
        if(param->bn_params != NULL)
            free(param->bn_params);
        if(param->bn_gradients != NULL)
            free(param->bn_gradients);
        if(param->bn_gradient_sums != NULL)
            free(param->bn_gradient_sums);
        if(param->bn_prev_gradients != NULL)
            free(param->bn_prev_gradients);
        if(param->bn_m_gradients != NULL)
            free(param->bn_m_gradients);
        if(param->bn_v_gradients != NULL)
            free(param->bn_v_gradients);
        if(param->pool2full != NULL)
            free(param->pool2full);
        if(param->col != NULL)
            free(param->col);
        if(param->multiplier != NULL)
            free(param->multiplier);
        if(param->sums != NULL)
            free(param->sums);
        free(param);
    }
}

void pcnn_model_update_layer(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i;
    float correction;
    float *weight_gradient_sums;
    float *bias_gradient_sums;
    const float scale = 1.0f / feeder->batch_size;

    /* Only convolution and fully-connected layers are updated. */
    if(layer->type != LAYER_TYPE_CONV && layer->type != LAYER_TYPE_FULL)
        return;

    if(queue->nproc > 1){
        weight_gradient_sums = layer->global_sumws;
        bias_gradient_sums = layer->global_sumbs;
    }
    else{
        weight_gradient_sums = layer->local_sumws;
        bias_gradient_sums = layer->local_sumbs;
    }

    if(model->optimizer == OPTIMIZER_SGD){
        cblas_saxpby(layer->num_gradients, model->weight_decay, layer->weight, 1, scale, weight_gradient_sums, 1);
        cblas_saxpby(layer->num_gradients, 1.0f, weight_gradient_sums, 1, model->momentum, layer->prev_sumws, 1);
        cblas_saxpy(layer->num_gradients, -1.0f * model->learning_rate, layer->prev_sumws, 1, layer->weight, 1);
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        /* update m */
        cblas_saxpby(layer->filter_size, (1.f - model->beta1) * scale, weight_gradient_sums, 1, model->beta1, layer->m_sumws, 1);

        /* update v */
#pragma omp parallel for
        for(i=0; i<layer->filter_size; i++)
            weight_gradient_sums[i] = powf(weight_gradient_sums[i], 2);
        cblas_saxpby(layer->filter_size, 1.f - model->beta2, weight_gradient_sums, 1, model->beta2, layer->v_sumws, 1);

        /* set updates */
#pragma omp parallel for
        for(i=0; i<layer->filter_size; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_sumws[i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<layer->filter_size; i++)
            param->col[i] = param->col[i] * layer->m_sumws[i];

        correction = sqrtf(1.f - param->beta2_decay) / (1.f - param->beta1_decay);
        cblas_saxpby(layer->filter_size, -1.f * model->learning_rate * correction, param->col, 1, 1.f, layer->weight, 1); 

        /* update m */
        cblas_saxpby(layer->bias_size, (1.f - model->beta1) * scale, bias_gradient_sums, 1, model->beta1, layer->m_sumbs, 1);

        /* update v */
#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            bias_gradient_sums[i] = powf(bias_gradient_sums[i], 2);
        cblas_saxpby(layer->bias_size, 1.f - model->beta2, bias_gradient_sums, 1, model->beta2, layer->v_sumbs, 1);

        /* set updates */
#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_sumbs[i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            param->col[i] = param->col[i] * layer->m_sumbs[i];

        cblas_saxpby(layer->bias_size, -1.f * model->learning_rate * correction, param->col, 1, 1.f, layer->bias, 1); 
    }

    if(layer->batch_norm)
        pcnn_bn_update(layer, model, param, feeder, queue);
}

void pcnn_model_partial_update_conv_layer(struct layer_t *layer,
                                          struct model_t *model,
                                          struct param_t *param,
                                          struct feeder_t *feeder,
                                          struct comm_queue_t *queue)
{
    int i, offset, length;
    float correction;
    float *gradients;
    const float scale = 1.0f / feeder->batch_size;

    /* Only convolution and fully-connected layers are updated. */
    if(layer->type != LAYER_TYPE_CONV && layer->type != LAYER_TYPE_FULL)
        return;

    if(queue->nproc > 1){
        if(layer->aligned_gradients){
            length = layer->num_local_gradients;
            offset = layer->num_local_gradients * queue->rank;
        }
        else{
            length = layer->scounts_gradients[queue->rank];
            offset = layer->sdispls_gradients[queue->rank];
        }
        gradients = layer->global_sumws;
    }
    else{
        length = layer->num_gradients;
        offset = 0;
        gradients = layer->local_sumws;
    }

    if(model->optimizer == OPTIMIZER_SGD){
        cblas_saxpby(length, model->weight_decay, &layer->weight[offset], 1, scale, gradients, 1);
        cblas_saxpby(length, 1.0f, gradients, 1, model->momentum, &layer->prev_sumws[offset], 1);
        cblas_saxpy(length, -1.0f * model->learning_rate, &layer->prev_sumws[offset], 1, &layer->weight[offset], 1);
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        /* update m */
        cblas_saxpby(length, (1.f - model->beta1) * scale, gradients, 1, model->beta1, &layer->m_sumws[offset], 1);

        /* update v */
#pragma omp parallel for
        for(i=0; i<length; i++)
            gradients[i] = powf(gradients[i], 2);
        cblas_saxpby(length, 1.f - model->beta2, gradients, 1, model->beta2, &layer->v_sumws[offset], 1);

        /* set updates */
#pragma omp parallel for
        for(i=0; i<length; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_sumws[offset + i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<length; i++)
            param->col[i] = param->col[i] * layer->m_sumws[offset + i];

        correction = sqrtf(1.f - param->beta2_decay) / (1.f - param->beta1_decay);
        cblas_saxpby(length, -1.f * model->learning_rate * correction, param->col, 1, 1.f, &layer->weight[offset], 1); 
    }

    /* Copy the locally updated model parameters to the send buffer so that
     * the local parameters are aggregated among all the processes. */
    memcpy(layer->local_sumws, &layer->weight[offset], sizeof(float) * length);

    if(layer->batch_norm)
        pcnn_bn_update(layer, model, param, feeder, queue);
}

void pcnn_model_partial_update_full_layer(struct layer_t *layer,
                                          struct model_t *model,
                                          struct param_t *param,
                                          struct feeder_t *feeder,
                                          struct comm_queue_t *queue)
{
    int i, length, offset;
    float correction;
    float *weight_gradient_sums;
    float *bias_gradient_sums;
    const float scale = 1.0f / feeder->batch_size;

    /* Only convolution and fully-connected layers are updated. */
    if(layer->type != LAYER_TYPE_CONV && layer->type != LAYER_TYPE_FULL)
        return;

    if(queue->nproc > 1){
        if(layer->aligned_weight){
            length = layer->local_weight_count;
            offset = layer->local_weight_count * queue->rank;
        }
        else{
            length = layer->scounts_weight[queue->rank];
            offset = layer->sdispls_weight[queue->rank];
        }
        weight_gradient_sums = layer->global_sumws;
        bias_gradient_sums = layer->global_sumbs;
    }
    else{
        length = layer->filter_size;
        offset = 0;
        weight_gradient_sums = layer->local_sumws;
        bias_gradient_sums = layer->local_sumbs;
    }

    if(model->optimizer == OPTIMIZER_SGD){
        /* Update weight parameters. */
        cblas_saxpby(length, model->weight_decay, &layer->weight[offset], 1, scale, weight_gradient_sums, 1);
        cblas_saxpby(length, 1.0f, weight_gradient_sums, 1, model->momentum, &layer->prev_sumws[offset], 1);
        cblas_saxpy(length, -1.0f * model->learning_rate, &layer->prev_sumws[offset], 1, &layer->weight[offset], 1);

        /* Update bias parameters. */
        cblas_saxpby(layer->bias_size, model->weight_decay, layer->bias, 1, scale, bias_gradient_sums, 1);
        cblas_saxpby(layer->bias_size, 1.0f, bias_gradient_sums, 1, model->momentum, layer->prev_sumbs, 1);
        cblas_saxpy(layer->bias_size, -1.0f * model->learning_rate, layer->prev_sumbs, 1, layer->bias, 1);
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        /* update m */
        cblas_saxpby(length, (1.f - model->beta1) * scale, weight_gradient_sums, 1, model->beta1, &layer->m_sumws[offset], 1);

        /* update v */
#pragma omp parallel for
        for(i=0; i<length; i++)
            weight_gradient_sums[i] = powf(weight_gradient_sums[i], 2);
        cblas_saxpby(length, 1.f - model->beta2, weight_gradient_sums, 1, model->beta2, &layer->v_sumws[offset], 1);

        /* set updates */
#pragma omp parallel for
        for(i=0; i<length; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_sumws[offset + i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<length; i++)
            param->col[i] = param->col[i] * layer->m_sumws[offset + i];

        correction = sqrtf(1.f - param->beta2_decay) / (1.f - param->beta1_decay);
        cblas_saxpby(length, -1.f * model->learning_rate * correction, param->col, 1, 1.f, &layer->weight[offset], 1); 

        /* update m */
        cblas_saxpby(layer->bias_size, (1.f - model->beta1) * scale, bias_gradient_sums, 1, model->beta1, layer->m_sumbs, 1);

        /* update v */
#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            bias_gradient_sums[i] = powf(bias_gradient_sums[i], 2);
        cblas_saxpby(layer->bias_size, 1.f - model->beta2, bias_gradient_sums, 1, model->beta2, layer->v_sumbs, 1);

        /* set updates */
#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_sumbs[i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<layer->bias_size; i++)
            param->col[i] = param->col[i] * layer->m_sumbs[i];

        cblas_saxpby(layer->bias_size, -1.f * model->learning_rate * correction, param->col, 1, 1.f, layer->bias, 1); 
    }

    /* Copy the locally updated model parameters to the send buffer so that
     * the local parameters are aggregated among all the processes. */
    memcpy(layer->local_sumws, &layer->weight[offset], sizeof(float) * length);

    if(layer->batch_norm)
        pcnn_bn_update(layer, model, param, feeder, queue);
}

void pcnn_model_get_default_features(struct feature_t *features)
{
    if(features == NULL){
        printf("[%s][%d] Invalid pointer is passed.\n", __FUNCTION__, __LINE__);
        return;
    }
    features->sub_type = 0;
    features->ReLU = 0;
    features->loss_type = LOSS_TYPE_NONE;
    features->batch_norm = 0;
    features->output_channels = 0;
    features->filter_depth = 1;
    features->filter_rows = 0;
    features->filter_cols = 0;
    features->num_channels = 0;
    features->image_depth = 1;
    features->image_rows = 0;
    features->image_cols = 0;
    features->pad_depth = 0;
    features->pad_rows = 0;
    features->pad_cols = 0;
    features->stride_depth = 1;
    features->stride_rows = 1;
    features->stride_cols = 1;
    features->mean = 0.0f;
    features->std = 0.0f;
    features->bottom_layer = -1;
    features->skip_from = -2;
    features->res_scale = 1.0f;
}

void pcnn_model_decay_learning_rate(struct model_t *model, struct param_t *param)
{
    if((param->num_trained_epochs % model->decay_steps == 0) &&
        (param->num_updates > 0)){
        printf("learning rate: %f -> %f\n", model->learning_rate, model->learning_rate * model->decay_factor);
        model->learning_rate *= model->decay_factor;
    }
}

void pcnn_model_init_comm_offsets(struct model_t *model, struct comm_queue_t *queue)
{
    int i;
    struct layer_t *layer;

    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];
        pcnn_model_calc_comm_offsets(layer, model, queue);
    }
}

void pcnn_model_put_momentum_together(struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    int i;
    struct layer_t *layer;
    struct comm_req_t req;

    if(queue->nproc == 1)
        return;

    /* post communications going through all the layers */
    for(i=0; i<model->num_layers; i++){
        layer = model->layers[i];

        if(layer->type == LAYER_TYPE_CONV){
            req.type = COMM_TYPE_GATHER_M0;
            req.layer_id = i;
            pcnn_comm_insert_req(model, queue, &req);

            pthread_mutex_lock(&queue->mut);
            while(queue->flag_gather_m0[i] == 1)
            pthread_cond_wait(&queue->cond, &queue->mut);
            pthread_mutex_unlock(&queue->mut);

            if(model->optimizer == OPTIMIZER_SGD){
                memcpy(layer->prev_sumws, layer->global_sumws, sizeof(float) * layer->num_gradients);
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                memcpy(layer->m_sumws, layer->global_sumws, sizeof(float) * layer->num_gradients);

                req.type = COMM_TYPE_GATHER_M1;
                req.layer_id = i;
                pcnn_comm_insert_req(model, queue, &req);

                pthread_mutex_lock(&queue->mut);
                while(queue->flag_gather_m1[i] == 1)
                pthread_cond_wait(&queue->cond, &queue->mut);
                pthread_mutex_unlock(&queue->mut);

                memcpy(layer->v_sumws, layer->global_sumws, sizeof(float) * layer->num_gradients);
            }
        }
        else if(layer->type == LAYER_TYPE_FULL){
            req.type = COMM_TYPE_GATHER_M0;
            req.layer_id = i;
            pcnn_comm_insert_req(model, queue, &req);

            pthread_mutex_lock(&queue->mut);
            while(queue->flag_gather_m0[i] == 1)
            pthread_cond_wait(&queue->cond, &queue->mut);
            pthread_mutex_unlock(&queue->mut);

            if(model->optimizer == OPTIMIZER_SGD){
                memcpy(layer->prev_sumws, layer->global_sumws, sizeof(float) * layer->filter_size);
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                memcpy(layer->m_sumws, layer->global_sumws, sizeof(float) * layer->filter_size);

                req.type = COMM_TYPE_GATHER_M1;
                req.layer_id = i;
                pcnn_comm_insert_req(model, queue, &req);

                pthread_mutex_lock(&queue->mut);
                while(queue->flag_gather_m1[i] == 1)
                pthread_cond_wait(&queue->cond, &queue->mut);
                pthread_mutex_unlock(&queue->mut);

                memcpy(layer->v_sumws, layer->global_sumws, sizeof(float) * layer->filter_size);
            }
        }
        else
            continue;
    }
}
