/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#ifdef USE_MKL 
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include "def.h"
#include "model.h"
#include "feeder.h"
#include "comm.h"
#include "conv.h"
#include "pool.h"
#include "full.h"
#include "upsample.h"
#include "relu.h"
#include "residual.h"
#include "batch_norm.h"
#include "loss.h"
#include "util.h"

/* Allreduce-based gradient averaging
 * This module implements the traditional allreduce-based data-parallelism.
 * All the gradients are averaged across the workers using allreduce.
 * This module supports both overlap and non-overlap. */

void pcnn_ffbp_allreduce_feedforward(int op, struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    int i;
    struct layer_t *top=NULL;
    struct layer_t *bottom=NULL;
    struct comm_req_t req;
    const float ratio = 1.f / queue->num_groups;

    /* Evaluate the training images, going through all the model layers. */
    for(i=0; i<model->num_layers; i++){
        top = model->layers[i];
        bottom = top->bottom_layer >= 0 ? model->layers[top->bottom_layer] : NULL;

        if(queue->nproc > 1){
            if(model->overlap == 1){
                if(op == OPERATION_TYPE_TRAINING && param->current_index > 0){
                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_reduce_g[i] == 1)
                        pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);

                    pcnn_model_update_layer(top, model, param, feeder, queue);
                }
            }
        }

        if(queue->nproc * queue->num_groups > 1){
            /* Average the parameters among communication groups. */
            if(top->type == LAYER_TYPE_CONV || top->type == LAYER_TYPE_FULL){
                if(queue->num_groups > 1 && param->num_updates % queue->sync_interval == 0){
                    req.type = COMM_TYPE_REDUCE_P;
                    req.layer_id = top->id;
                    pcnn_comm_insert_req(model, queue, &req);

                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_reduce_p[i] == 1)
                    pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);

                    cblas_saxpby(top->num_gradients, ratio, top->global_sumws, 1, 0, top->weight, 1);
                }
            }
        }

        if(top->type == LAYER_TYPE_CONV){
            pcnn_conv_ff(op, feeder->local_batch_size, bottom, top, feeder, param);

            if(top->batch_norm)
                pcnn_bn_ff(op, top, model, param, feeder);
        }
        else if(top->type == LAYER_TYPE_POOL){
            pcnn_pool_ff(op, feeder->local_batch_size, bottom, top, param);
        }
        else if(top->type == LAYER_TYPE_FULL){
            pcnn_full_ff(op, feeder->local_batch_size, bottom, top, param);
        }
        else if(top->type == LAYER_TYPE_UPSAMPLE){
            pcnn_upsample_ff(feeder->local_batch_size, model->upsample_ratio, bottom, top);
        }

        if(top->skip_from >= -1)// when skip_from is -1, the input images are directly added to the activations
            pcnn_residual_ff(top, model, feeder, queue);

        if(top->ReLU)
            pcnn_relu_ff(top, model, feeder);
    }

    /* The last activation function (softmax/MSE). */
    pcnn_loss_ff(top, model, feeder);

    /* Check the current accuracy. */
    pcnn_util_evaluate(model, param, feeder, queue);

    /* When running validation, we calculate loss here. */
    if(op == OPERATION_TYPE_VALIDATION)
        pcnn_loss_bp(top, model, param, feeder);
}

void pcnn_ffbp_allreduce_backprop(int op, struct feeder_t *feeder, struct model_t * model, struct param_t *param, struct comm_queue_t *queue)
{
    int i=0;
    struct layer_t *top=NULL;
    struct layer_t *bottom=NULL;
    struct comm_req_t req;

    /* First, calculate the errors at the output layer.
     * The locally calculated training loss is accumulated until
     * the end of the epoch. */
    top = model->layers[model->num_layers-1];
    pcnn_loss_bp(top, model, param, feeder);

    /* Backpropagate the errors going through the fully-connected layers 
    * and compute the gradients. */
    for(i=model->num_layers-1; i>=0; i--){
        top = model->layers[i];
        bottom = top->bottom_layer >= 0 ? model->layers[top->bottom_layer] : NULL;

        if(top->ReLU)
            pcnn_relu_bp(top, model, feeder);

        if(top->skip_from >= 0)
            pcnn_residual_bp(top, model, feeder);

        if(top->type == LAYER_TYPE_FULL){
            pcnn_full_bp(op, feeder->local_batch_size, bottom, top, param);
            pcnn_full_gradw_pattern0(bottom, top, model, param, feeder);
            pcnn_full_gradb_pattern0(top, model, feeder);

            if(queue->nproc > 1 && model->overlap == 1){
                req.type = COMM_TYPE_REDUCE_G;
                req.layer_id = i;
                pcnn_comm_insert_req(model, queue, &req);
            }
        }
        else if(top->type == LAYER_TYPE_POOL){
            pcnn_pool_bp(feeder->local_batch_size, bottom, top);
        }
        else if(top->type == LAYER_TYPE_CONV){
            if(top->batch_norm)
                pcnn_bn_bp(top, model, param, feeder);

            pcnn_conv_bp(op, feeder->local_batch_size, bottom, top, param);
            pcnn_conv_gradw(op, feeder->local_batch_size, bottom, top, feeder, param);
            pcnn_conv_gradb(top, model, param, feeder);

            if(queue->nproc > 1 && model->overlap == 1){
                req.type = COMM_TYPE_REDUCE_G;
                req.layer_id = i;
                pcnn_comm_insert_req(model, queue, &req);
            }
        }
        else if(top->type == LAYER_TYPE_UPSAMPLE){
            pcnn_upsample_bp(feeder->local_batch_size, model->upsample_ratio, bottom, top);
        }
    }

    if(queue->nproc> 1 && model->overlap == 0){
        /* Initiate all the communications at once. */
        for(i=model->num_layers-1; i>=0; i--){
            top = model->layers[i];
            if(top->type == LAYER_TYPE_CONV || top->type == LAYER_TYPE_FULL){
                req.type = COMM_TYPE_REDUCE_G;
                req.layer_id = i;
                pcnn_comm_insert_req(model, queue, &req);
            }
        }
    }
}

void pcnn_ffbp_allreduce_update(struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i;

    if(queue->nproc == 1){
        for(i=0; i<model->num_layers; i++)
			pcnn_model_update_layer(model->layers[i], model, param, feeder, queue);
    }
    else{
        if(model->overlap == 0){
            /* Without overlapping, wait until all the communications are done. 
             * Then, update the entire model at once. */
            for(i=0; i<model->num_layers; i++){
                pthread_mutex_lock(&queue->mut);
                while(queue->flag_reduce_g[i] == 1)
                    pthread_cond_wait(&queue->cond, &queue->mut);
                pthread_mutex_unlock(&queue->mut);
            }

            for(i=0; i<model->num_layers; i++)
                pcnn_model_update_layer(model->layers[i], model, param, feeder, queue);
        }
        else{
            /* If the overlapping is turnned on, update the model here 
             * when processing the last mini-batch. For other mini-batches,
             * the parameters are updated in the next iteration feedforward. */
            if((param->current_index + (feeder->batch_size * queue->num_groups)) >= feeder->num_train_images){
                for(i=model->num_layers-1; i>=0; i--){
                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_reduce_g[i] == 1)
                        pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);

                    pcnn_model_update_layer(model->layers[i], model, param, feeder, queue);
                }
            }
        }
    }
    param->num_updates++;
}
