/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "def.h"
#include "model.h"
#include "feeder.h"

/* static functions */
static void pcnn_softmax_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder);
static void pcnn_softmax_loss(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_softmax_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_mae_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_mse_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);

void pcnn_loss_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder)
{
    if(layer->loss_type == LOSS_TYPE_SOFTMAX)
        pcnn_softmax_ff(layer, model, feeder);

    /* Other loss functions do not have any computation at feed-forward. */
}

void pcnn_loss_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    if(layer->loss_type == LOSS_TYPE_SOFTMAX)
        pcnn_softmax_bp(layer, model, param, feeder);
    else if(layer->loss_type == LOSS_TYPE_MAE)
        pcnn_mae_bp(layer, model, param, feeder);
    else if(layer->loss_type == LOSS_TYPE_MSE)
        pcnn_mse_bp(layer, model, param, feeder);
    else
        printf("[%s][%d] loss function is called for non-output layer.\n", __FUNCTION__, __LINE__);
}

static void pcnn_softmax_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder)
{
    int i = 0, j = 0;
    float max = 0.0f;
    float sum = 0.0f;

#pragma omp parallel for private(j, max, sum)
    for(i=0; i<feeder->local_batch_size; i++){
        /* Find the maximum output from the layer. */
        max = 0;
        for(j=0; j<layer->num_neurons; j++){
            if(max < layer->a[j * feeder->local_batch_size + i])
                max = layer->a[j * feeder->local_batch_size + i];
        }

        /* Subtract the max from each output in case exp() exceeds the range. */
        sum = 0;
        for(j=0; j<layer->num_neurons; j++){
            layer->a[j * feeder->local_batch_size + i] -= max;
            sum += expf(layer->a[j * feeder->local_batch_size + i]);
        }

        /* Calculate the activations. */
        for(j=0; j<layer->num_neurons; j++){
            layer->a[j * feeder->local_batch_size + i] = expf(layer->a[j * feeder->local_batch_size + i]) / sum;
        }
    }
}

static void pcnn_softmax_loss(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i = 0, j = 0;
    int max = 0, label_offset = 0;
    float loss = 0.0f;
    
#pragma omp parallel for private(j, max, label_offset) reduction(+:loss)
    for(i=0; i<feeder->local_batch_size; i++){
        label_offset = i * feeder->label_size;
        for(j=0; j<layer->num_neurons; j++){
            if(feeder->label[label_offset++] == 1)
                break;
        }
        max = j * feeder->local_batch_size + i;
        loss -= logf(layer->a[max]);
    }
    param->local_loss += (loss / feeder->local_batch_size);
}

static void pcnn_softmax_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i = 0, j = 0;
    int offset = 0, label_offset = 0;

#pragma omp parallel for private(j, offset, label_offset)
    for(i=0; i<feeder->local_batch_size; i++){
        label_offset = i * layer->num_neurons;
        for(j=0; j<layer->num_neurons; j++){
            offset = j * feeder->local_batch_size + i;
            layer->e[offset] = layer->a[offset] - feeder->label[label_offset+j];
        }
    }

    pcnn_softmax_loss(layer, model, param, feeder);
}

static void pcnn_mse_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i, j, k, l;
    int area;
    int src_off;
    int dst_off;
    float sum = 0;

    if(layer->type == LAYER_TYPE_CONV || layer->type == LAYER_TYPE_UPSAMPLE){
        area = layer->output_rows * layer->output_cols;
#pragma omp parallel for private(j, k, l, src_off, dst_off) reduction(+:sum)
        for(i=0; i<layer->output_channels; i++){
            for(j=0; j<feeder->local_batch_size; j++){
                src_off = (j * layer->output_channels + i) * area;
                dst_off = (i * feeder->local_batch_size + j) * area;
                for(k=0; k<layer->output_rows; k++){
                    for(l=0; l<layer->output_cols; l++){
                        layer->e[dst_off] = layer->a[dst_off] - feeder->label[src_off];
                        sum += powf(layer->e[dst_off], 2);
                        dst_off++;
                        src_off++;
                    }
                }
            }
        }
    }
    else if(layer->type == LAYER_TYPE_FULL){
#pragma omp parallel for private(j, src_off, dst_off) reduction(+:sum)
        for(i=0; i<feeder->local_batch_size; i++){
            src_off = i * layer->num_neurons;
            for(j=0; j<layer->num_neurons; j++){
                dst_off = j * feeder->local_batch_size + i;
                layer->e[dst_off] = layer->a[dst_off] - feeder->label[src_off++];
                sum += powf(layer->e[dst_off], 2);
            }
        }
    }
    param->local_loss += (sum / (layer->num_neurons * feeder->local_batch_size));
}

static void pcnn_mae_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i, j, k, l;
    int area;
    int src_off;
    int dst_off;
    float sum = 0;

    if(layer->type == LAYER_TYPE_CONV || layer->type == LAYER_TYPE_UPSAMPLE){
        area = layer->output_rows * layer->output_cols;
#pragma omp parallel for private(j, k, l, src_off, dst_off) reduction(+:sum)
        for(i=0; i<layer->output_channels; i++){
            for(j=0; j<feeder->local_batch_size; j++){
                src_off = (j * layer->output_channels + i) * area;
                dst_off = (i * feeder->local_batch_size + j) * area;
                for(k=0; k<layer->output_rows; k++){
                    for(l=0; l<layer->output_cols; l++){
                        if(layer->a[dst_off] > feeder->label[src_off])
                            layer->e[dst_off] = 1.0f;
                        else if(layer->a[dst_off] < feeder->label[src_off])
                            layer->e[dst_off] = -1.0f;
                        else
                            layer->e[dst_off] = 0.f;

                        sum += fabsf(layer->a[dst_off] - feeder->label[src_off]);
                        dst_off++;
                        src_off++;
                    }
                }
            }
        }
    }
    else if(layer->type == LAYER_TYPE_FULL){
#pragma omp parallel for private(j, src_off, dst_off) reduction(+:sum)
        for(i=0; i<feeder->local_batch_size; i++){
            src_off = i * layer->num_neurons;
            for(j=0; j<layer->num_neurons; j++){
                dst_off = j * feeder->local_batch_size + i;
                if(layer->a[dst_off] > feeder->label[src_off])
                    layer->e[dst_off] = 1.0f;
                else if(layer->a[dst_off] < feeder->label[src_off])
                    layer->e[dst_off] = -1.0f;
                else
                    layer->e[dst_off] = 0.f;

                sum += fabsf(layer->a[dst_off] - feeder->label[src_off]);
                dst_off++;
                src_off++;
            }
        }
    }
    param->local_loss += (sum / (layer->num_neurons * feeder->local_batch_size));
}
