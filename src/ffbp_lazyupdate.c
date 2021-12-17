/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <string.h>
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
#include "transform.h"

void pcnn_ffbp_lazy_backprop(int op, struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    int i=0, j=0;
    struct comm_req_t req;
    struct layer_t *top = NULL;
    struct layer_t *bottom = NULL;

    /* First, calculate the errors at the output layer.
     * The locally calculated training loss is accumulated until
     * the end of the epoch. */
    top = model->layers[model->num_layers-1];
    pcnn_loss_bp(top, model, param, feeder);

    /* From the top to the bottom, compute the errors and propagate back. */
    for(i=model->num_layers - 1; i>=0; i--){
        top = model->layers[i];
        bottom = top->bottom_layer >= 0 ? model->layers[top->bottom_layer] : NULL;

        if(top->ReLU)
            pcnn_relu_bp(top, model, feeder);

        /* The errors at top should be also propagated into the branch. */
        if(top->skip_from >= 0)
            pcnn_residual_bp(top, model, feeder);

        if(top->type == LAYER_TYPE_CONV){
            if(top->batch_norm)
                pcnn_bn_bp(top, model, param, feeder);
            pcnn_conv_bp(op, feeder->local_batch_size, bottom, top, param);

            /* Compute gradients at convolution layers. 
             * alltoall communications are initiated at each layer. */
            pcnn_conv_gradw(op, feeder->local_batch_size, bottom, top, feeder, param);
            pcnn_conv_gradb(top, model, param, feeder);
            if(queue->nproc > 1){
                if(top->sub_type == 0){
                    req.type = COMM_TYPE_ALL2ALL_G;
                    req.layer_id = top->id;
                    pcnn_comm_insert_req(model, queue, &req);
                }
                else{
                    /* Lazy update layers just accumulate the gradients. 
                     * This computation time may overlap the previous communications. */
                    cblas_saxpby(top->num_gradients, 1, top->local_sumws, 1.f, 1, top->local_accum, 1.f);
                }
            }
            else{
                if(top->sub_type == 1)
                    cblas_saxpby(top->num_gradients, 1, top->local_sumws, 1.f, 1, top->local_accum, 1.f);
            }
        }
        else if(top->type == LAYER_TYPE_POOL){
            pcnn_pool_bp(feeder->local_batch_size, bottom, top);
        }
        else if(top->type == LAYER_TYPE_FULL){
            if(queue->nproc > 1){
                req.type = COMM_TYPE_GATHER_E;
                req.layer_id = top->id;
                pcnn_comm_insert_req(model, queue, &req);
            }
            pcnn_full_bp(op, feeder->local_batch_size, bottom, top, param);
        }
        else if(top->type == LAYER_TYPE_UPSAMPLE){
            pcnn_upsample_bp(feeder->local_batch_size, model->upsample_ratio, bottom, top);
        }
    }
    param->num_accumulated++;

    if(queue->nproc > 1){
        /* lazy update
         * A single allreduce is posted before the second step communications of the regular conv layers.
         * This is important because the second step communications should overlap with the next forward computations. */
        if(param->num_accumulated == param->interval){
            for(i=0; i<model->b; i++){
                top = model->layers[i];
                if(top->type == LAYER_TYPE_CONV && top->sub_type == 1){
                    req.type = COMM_TYPE_REDUCE_G;
                    req.layer_id = top->id;
                    pcnn_comm_insert_req(model, queue, &req);
                }
            }
            req.type = COMM_TYPE_REDUCE_AG;
            pcnn_comm_insert_req(model, queue, &req);
        }

        /* Wait until the first step communications (all-to-all) are
         * finished at all the convolution layers. Here, we assume that
         * the bottom layer is a convolution layer. */
        pthread_mutex_lock(&queue->mut);
        while(queue->flag_all2all_g[model->b - 1] == 1)
            pthread_cond_wait(&queue->cond, &queue->mut);
        pthread_mutex_unlock(&queue->mut);

        /* Then, compute the global gradient sums and
         * update the model parameters. Note it traverses over
         * all the layers from the bottom to the top so that
         * the final step communications (allgather) are overlapped with
         * the next forward computation. */
        for(i=0; i<model->num_layers; i++){
            top = model->layers[i];
            if(top->type == LAYER_TYPE_CONV){
                if(top->sub_type == 0){// regular layers only
                    for(j=1; j<queue->nproc; j++)
                        cblas_saxpy(top->num_local_gradients, 1, &top->global_sumws[j * top->num_local_gradients], 1, top->global_sumws, 1);
                    pcnn_model_partial_update_conv_layer(top, model, param, feeder, queue);

                    req.type = COMM_TYPE_GATHER_CONV_PARAM;
                    req.layer_id = top->id;
                    pcnn_comm_insert_req(model, queue, &req);
                }
            }
        }
    }

    /* Convolution layers are done. Now work on fully-connected layers.
     * First, transform the data layout of the scattered activations and
     * gathered errors. Then, compute the gradients, update the local 
     * model parameters, and initiate the last allgather for the updated
     * local model parameters. */
    if(param->first_full_id > -1){
        for(i=param->first_full_id; i<model->num_layers; i++){
            top = model->layers[i];
            bottom = (i == 0)?NULL:model->layers[i-1];

            if(top->type == LAYER_TYPE_FULL){
                /* Wait until the activations are ready to use and transform the data structure
                 * so that they can be multiplied with errors. */
                if(queue->nproc > 1){
                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_all2all_a[bottom->id] == 1)
                        pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);

                    pcnn_transform_rearrange(bottom->recv_a, bottom->rep_a, 
                                             bottom->num_neurons, 
                                             feeder->batch_size, 
                                             queue->nproc);

                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_gather_e[top->id] == 1)
                        pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);

                    pcnn_transform_rearrange(top->recv_e, top->rep_e, 
                                             top->num_neurons * queue->nproc, 
                                             feeder->batch_size, 
                                             queue->nproc);
                }
                /* In pattern 1, first calculate the gradients first
                 * and update the partial model and then aggregate the
                 * entire model parameters across the workers. */
                pcnn_full_gradw_pattern1(bottom, top, model, param, feeder, queue);
                pcnn_full_gradb_pattern1(top, model, feeder, queue);

                if(queue->nproc > 1){
                    pcnn_model_partial_update_full_layer(top, model, param, feeder, queue);

                    req.type = COMM_TYPE_GATHER_W;
                    req.layer_id = top->id;
                    pcnn_comm_insert_req(model, queue, &req);
                }
            }
        }
    }

    /* lazy update
     * Update the parameters with the accumulated gradients. */
    if(param->num_accumulated >= param->interval){
        /* Wait until the accumulated gradients are reduced. Then, update the interval first. */
        pthread_mutex_lock(&queue->mut);
        while(queue->flag_reduce_ag == 1)
            pthread_cond_wait(&queue->cond, &queue->mut);
        pthread_mutex_unlock(&queue->mut);
        pcnn_model_update_interval_layer(model->b - 1, model, param, queue);

        /* Then, update the parameters and initialize the accum buffers. */
        for(i=0; i<model->b; i++){
            top = model->layers[i];
            if(top->type == LAYER_TYPE_CONV && top->sub_type == 1){
                if(queue->nproc > 1)
                    memcpy(top->global_sumws, top->global_accum, sizeof(float) * top->num_gradients);
                else
                    memcpy(top->local_sumws, top->local_accum, sizeof(float) * top->num_gradients);
                memset(top->local_accum, 0, sizeof(float) * top->num_gradients);
                pcnn_model_update_layer(top, model, param, feeder, queue);
            }
        }
        param->num_accumulated = 0;
        param->num_lazy_updates++;
    }
}

void pcnn_ffbp_lazy_feedforward(int op, struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
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

        /* Wait until parameters are aggregated. */
        if(queue->nproc > 1){
            pthread_mutex_lock(&queue->mut);
            while(queue->flag_gather_g[i] == 1)
                pthread_cond_wait(&queue->cond, &queue->mut);
            pthread_mutex_unlock(&queue->mut);
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
            if(queue->nproc > 1 && op == OPERATION_TYPE_TRAINING){
                if(model->overlap == 1){
                    req.type = COMM_TYPE_ALL2ALL_A;
                    req.layer_id = bottom->id;
                    pcnn_comm_insert_req(model, queue, &req);
                }
            }
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

void pcnn_ffbp_lazy_update(struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
	int i;

	if(queue->nproc == 1){
		for(i=0; i<model->num_layers; i++)
			pcnn_model_update_layer(model->layers[i], model, param, feeder, queue);
	}
	else{
        if(model->overlap == 0){
            /* Wait until all the gradient sums are ready. */
            for(i=0; i<model->num_layers; i++){
                pthread_mutex_lock(&queue->mut);
                while(queue->flag_gather_g[i] == 1)
                    pthread_cond_wait(&queue->cond, &queue->mut);
                pthread_mutex_unlock(&queue->mut);
            }
        }
        else{
            /* If overlapping is turnned on, wait the communications here
             * only for the last mini-batch. For other mini-batches, the communications
             * are overlapped with the next iteration feed-forward computation time. */
            if((param->current_index + (feeder->batch_size * queue->num_groups)) >= feeder->num_train_images){
                printf("The last batch... wait here\n");
                for(i=0; i<model->num_layers; i++){
                    pthread_mutex_lock(&queue->mut);
                    while(queue->flag_gather_g[i] == 1)
                        pthread_cond_wait(&queue->cond, &queue->mut);
                    pthread_mutex_unlock(&queue->mut);
                }
            }
        }
	}

	param->num_updates++;
}
