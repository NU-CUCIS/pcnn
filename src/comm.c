/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include "def.h"
#include "config.h"
#include "model.h"
#include "comm.h"
#include "feeder.h"

/* This is a communication module.
 * In order to overlap the communications with the computations,
 * we employ a communication-dedicated pthread per process.
 * The thread calls blocking MPI operations whenever there is
 * a request from the main thread. */
struct comm_queue_t *pcnn_comm_init(int num_groups, int sync_interval)
{
    struct comm_queue_t *queue;
    int rank, nproc;
    int color, key;
    int group_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if((num_groups > nproc) ||
       (num_groups <= 0) ||
       (nproc % num_groups != 0)){
        printf("[%s][%s][%d] Invalid number of communication groups!\n", __FILE__, __FUNCTION__, __LINE__);
        return NULL;
    }

    queue = (struct comm_queue_t *)malloc(sizeof(struct comm_queue_t));
    queue->queue_size = 32; // The message queue will keep up to 32 messages.
    queue->queue = (struct comm_req_t *)malloc(sizeof(struct comm_req_t) * queue->queue_size);
    queue->req_count = 0;
    queue->num_groups = num_groups;
    queue->sync_interval = sync_interval;
    queue->world = MPI_COMM_WORLD;
    MPI_Comm_rank(queue->world, &queue->global_rank);
    MPI_Comm_size(queue->world, &queue->global_nproc);

    /* The flags for synchronizing between communication thread and 
     * computation thread. */
    queue->flag_reduce = 0;
    queue->flag_reduce_l = 0;
    queue->flag_reduce_ag = 0;
    queue->flag_reduce_g = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_reduce_p = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_gather_e = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_gather_g = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_all2all_a = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_all2all_g = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_gather_m0 = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));
    queue->flag_gather_m1 = (int *)calloc(MAX_NUM_LAYERS, sizeof(int));

    pthread_cond_init(&queue->cond, NULL);
    pthread_mutex_init(&queue->mut, NULL);

    /* If the number of groups is larger than 1, each group performs
     * localSGD training and the model parameters are periodically
     * averaged among all the groups. Here, the MPI communicator is
     * split based on the requested number of groups. */
    group_size = nproc / num_groups;
    color = rank / group_size;
    key = rank % group_size;
    queue->group_id = color;

    MPI_Comm_split(MPI_COMM_WORLD, color, key, &queue->comm);
    MPI_Comm_rank(queue->comm, &queue->rank);
    MPI_Comm_size(queue->comm, &queue->nproc);

    color = rank % group_size;
    key = rank / group_size;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &queue->across);

    return queue;
}

void pcnn_comm_destroy(struct comm_queue_t *queue)
{
    if(queue != NULL){
        if(queue->queue != NULL)
            free(queue->queue);
        if(queue->flag_reduce_g != NULL)
            free(queue->flag_reduce_g);
        if(queue->flag_reduce_p != NULL)
            free(queue->flag_reduce_p);
        if(queue->flag_gather_e != NULL)
            free(queue->flag_gather_e);
        if(queue->flag_gather_g != NULL)
            free(queue->flag_gather_g);
        if(queue->flag_all2all_a != NULL)
            free(queue->flag_all2all_a);
        if(queue->flag_all2all_g != NULL)
            free(queue->flag_all2all_g);
        if(queue->flag_gather_m0 != NULL)
            free(queue->flag_gather_m0);
        if(queue->flag_gather_m1 != NULL)
            free(queue->flag_gather_m1);
        pthread_cond_destroy(&queue->cond);
        pthread_mutex_destroy(&queue->mut);
        MPI_Comm_free(&queue->comm);
        MPI_Comm_free(&queue->across);
        free(queue);
    }
    return;
}

int pcnn_comm_insert_req(struct model_t *model, struct comm_queue_t *queue, struct comm_req_t *req)
{
    int offset=0;

    pthread_mutex_lock(&queue->mut);
    /* If queue is full, just wait until the requests are serviced. 
    */
    while(queue->req_count >= queue->queue_size){
        pthread_cond_wait(&queue->cond, &queue->mut);
    }

    offset = (queue->index + queue->req_count) % queue->queue_size;

    queue->queue[offset].type = req->type;
    queue->queue[offset].layer_id = req->layer_id;
    queue->req_count += 1;

    if(req->type == COMM_TYPE_REDUCE)
        queue->flag_reduce = 1;
    else if(req->type == COMM_TYPE_REDUCE_G)
        queue->flag_reduce_g[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_REDUCE_P)
        queue->flag_reduce_p[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_REDUCE_L)
        queue->flag_reduce_l = 1;
    else if(req->type == COMM_TYPE_REDUCE_CONV_PARAM)
        queue->flag_reduce_p[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_REDUCE_FULL_PARAM)
        queue->flag_reduce_p[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_GATHER_E)
        queue->flag_gather_e[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_ALL2ALL_A)
        queue->flag_all2all_a[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_ALL2ALL_G)
        queue->flag_all2all_g[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_GATHER_W)
        queue->flag_gather_g[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_GATHER_CONV_PARAM)
        queue->flag_gather_g[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_GATHER_M0)
        queue->flag_gather_m0[req->layer_id] = 1;
    else if(req->type == COMM_TYPE_GATHER_M1)
        queue->flag_gather_m1[req->layer_id] = 1;

    pthread_mutex_unlock(&queue->mut);
    pthread_cond_signal(&queue->cond);

    return 0;
}

void *pcnn_comm_thread(void *ptr)
{
	int type, layer_id;
    int length, offset;
	float *send_buf;
	struct msg_t *msg;
	struct model_t *model;
	struct param_t *param;
    struct feeder_t *feeder;
	struct layer_t *layer;
	struct comm_queue_t *queue;

    /* CPU Affinity setting */
    int ret;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(67, &cpuset);
    ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if(ret != 0){
        printf("[%s][%d] pthread_set_affinity_np failed! Your parallel training may suffer from more expensive communictions...\n", __FUNCTION__, __LINE__);
    }

	msg = (struct msg_t *)ptr;
	model = (struct model_t *)msg->model;
	param = (struct param_t *)(msg->param);
    feeder = (struct feeder_t *)(msg->feeder);
	queue = (struct comm_queue_t *)(msg->queue);
	queue->index = 0;

	while(1){
/* Dequeue a request from the request queue. */
        pthread_mutex_lock(&queue->mut);
        while(queue->req_count == 0){
            pthread_cond_wait(&queue->cond, &queue->mut);
        }

        type = queue->queue[queue->index].type;
        layer_id = queue->queue[queue->index].layer_id;

        queue->req_count--;
        queue->index++;
        if(queue->index == queue->queue_size)
            queue->index = 0;

        pthread_mutex_unlock(&queue->mut);
        pthread_cond_signal(&queue->cond);

/* Actual blocking MPI communication below.
 * Various REDUCE / GATHER / ALL2ALL are supported. */
        if(type == COMM_TYPE_REDUCE){
            MPI_Allreduce(param->gradients, 
                          param->gradient_sums, 
                          param->total_size,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->comm);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_REDUCE_G){
            layer = model->layers[layer_id];

            MPI_Allreduce(layer->local_sumws, 
                          layer->global_sumws, 
                          layer->num_gradients,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->comm);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce_g[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_REDUCE_P){
            layer = model->layers[layer_id];

            MPI_Allreduce(layer->weight, 
                          layer->global_sumws, 
                          layer->num_gradients,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->across);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce_p[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_REDUCE_CONV_PARAM){
            layer = model->layers[layer_id];

            if(layer->aligned_gradients){
                length = layer->num_local_gradients;
                offset = layer->num_local_gradients * queue->rank;
            }
            else{
                length = layer->scounts_gradients[queue->rank];
                offset = layer->sdispls_gradients[queue->rank];
            }
            MPI_Allreduce(&layer->weight[offset], 
                          layer->global_sumws, 
                          length,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->across);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce_p[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_REDUCE_FULL_PARAM){
            layer = model->layers[layer_id];

            if(layer->aligned_weight){
                length = layer->local_weight_count;
                offset = layer->local_weight_count * queue->rank;
            }
            else{
                length = layer->scounts_weight[queue->rank];
                offset = layer->sdispls_weight[queue->rank];
            }
            MPI_Allreduce(&layer->weight[offset], 
                          layer->global_sumws, 
                          length,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->across);

            MPI_Allreduce(layer->bias, 
                          layer->global_sumbs, 
                          layer->bias_size,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->across);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce_p[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_REDUCE_L){
            MPI_Allreduce(&param->local_loss,
                          &param->global_loss,
                          1,
                          MPI_FLOAT, 
                          MPI_SUM, 
                          queue->comm);

            pthread_mutex_lock(&queue->mut);
            queue->flag_reduce_l = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_GATHER_E){
            layer = model->layers[layer_id];
            length = layer->num_neurons * feeder->local_batch_size;

            MPI_Allgather(layer->e,
                          length,
                          MPI_FLOAT,
                          layer->recv_e,
                          length,
                          MPI_FLOAT, 
                          queue->comm);

            pthread_mutex_lock(&queue->mut);
            queue->flag_gather_e[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_ALL2ALL_A){
            layer = model->layers[layer_id];
            send_buf = (layer->type == LAYER_TYPE_FULL) ? layer->a : param->pool2full;
            length = layer->num_neurons * feeder->local_batch_size / queue->nproc;

            MPI_Alltoall(send_buf,
                         length,
                         MPI_FLOAT,
                         layer->recv_a,
                         length,
                         MPI_FLOAT, 
                         queue->comm);

            pthread_mutex_lock(&queue->mut);
            queue->flag_all2all_a[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_ALL2ALL_G){
            layer = model->layers[layer_id];

            if(layer->aligned_gradients == 0){
                MPI_Alltoallv(layer->local_sumws,
                              layer->scounts_gradients,
                              layer->sdispls_gradients,
                              MPI_FLOAT,
                              layer->global_sumws,
                              layer->rcounts_gradients,
                              layer->rdispls_gradients,
                              MPI_FLOAT,
                              queue->comm);
            }
            else{
                MPI_Alltoall(layer->local_sumws,
                             layer->num_local_gradients,
                             MPI_FLOAT,
                             layer->global_sumws,
                             layer->num_local_gradients,
                             MPI_FLOAT, 
                             queue->comm);
            }

            pthread_mutex_lock(&queue->mut);
            queue->flag_all2all_g[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_GATHER_W){
            layer = model->layers[layer_id];
            length = layer->local_weight_count;

            if(layer->aligned_weight == 0){
                MPI_Allgatherv(layer->local_sumws,
                               length,
                               MPI_FLOAT,
                               layer->weight,
                               layer->scounts_weight,
                               layer->sdispls_weight,
                               MPI_FLOAT,
                               queue->comm);
            }
            else{
                MPI_Allgather(layer->local_sumws,
                              length,
                              MPI_FLOAT,
                              layer->weight,
                              length,
                              MPI_FLOAT, 
                              queue->comm);
            }

            pthread_mutex_lock(&queue->mut);
            queue->flag_gather_g[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_GATHER_CONV_PARAM){
            layer = model->layers[layer_id];
            length = layer->num_local_gradients;

            if(layer->aligned_gradients == 0){
                MPI_Allgatherv(layer->local_sumws,
                               length,
                               MPI_FLOAT,
                               layer->weight,
                               layer->scounts_gradients,
                               layer->sdispls_gradients,
                               MPI_FLOAT,
                               queue->comm);
            }
            else{
                MPI_Allgather(layer->local_sumws,
                              length,
                              MPI_FLOAT,
                              layer->weight,
                              length,
                              MPI_FLOAT, 
                              queue->comm);
            }

            pthread_mutex_lock(&queue->mut);
            queue->flag_gather_g[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_GATHER_M0){
            layer = model->layers[layer_id];
            if(layer->type == LAYER_TYPE_CONV){
                if(layer->aligned_gradients){
                    length = layer->num_local_gradients;
                    offset = layer->num_local_gradients * queue->rank;
                }
                else{
                    length = layer->scounts_gradients[queue->rank];
                    offset = layer->sdispls_gradients[queue->rank];
                }

                if(model->optimizer == OPTIMIZER_SGD){
                    if(layer->aligned_gradients){
                        MPI_Allgather(&layer->prev_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->prev_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_gradients,
                                       layer->sdispls_gradients,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
                else if(model->optimizer == OPTIMIZER_ADAM){
                    if(layer->aligned_weight){
                        MPI_Allgather(&layer->m_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->m_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_gradients,
                                       layer->sdispls_gradients,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
            }
            else if(layer->type == LAYER_TYPE_FULL){
                if(layer->aligned_weight){
                    length = layer->local_weight_count;
                    offset = layer->local_weight_count * queue->rank;
                }
                else{
                    length = layer->scounts_weight[queue->rank];
                    offset = layer->sdispls_weight[queue->rank];
                }

                if(model->optimizer == OPTIMIZER_SGD){
                    if(layer->aligned_weight){
                        MPI_Allgather(&layer->prev_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->prev_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_weight,
                                       layer->sdispls_weight,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
                else if(model->optimizer == OPTIMIZER_ADAM){
                    if(layer->aligned_weight){
                        MPI_Allgather(&layer->m_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->m_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_weight,
                                       layer->sdispls_weight,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
            }

            pthread_mutex_lock(&queue->mut);
            queue->flag_gather_m0[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_GATHER_M1){
            layer = model->layers[layer_id];
            if(layer->type == LAYER_TYPE_CONV){
                if(layer->aligned_gradients){
                    length = layer->num_local_gradients;
                    offset = layer->num_local_gradients * queue->rank;
                }
                else{
                    length = layer->scounts_gradients[queue->rank];
                    offset = layer->sdispls_gradients[queue->rank];
                }

                if(model->optimizer == OPTIMIZER_ADAM){
                    if(layer->aligned_gradients){
                        MPI_Allgather(&layer->v_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->v_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_gradients,
                                       layer->sdispls_gradients,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
            }
            else if(layer->type == LAYER_TYPE_FULL){
                if(layer->aligned_weight){
                    length = layer->local_weight_count;
                    offset = layer->local_weight_count * queue->rank;
                }
                else{
                    length = layer->scounts_weight[queue->rank];
                    offset = layer->sdispls_weight[queue->rank];
                }

                if(model->optimizer == OPTIMIZER_ADAM){
                    if(layer->aligned_weight){
                        MPI_Allgather(&layer->v_sumws[offset],
                                      length,
                                      MPI_FLOAT,
                                      layer->global_sumws,
                                      length,
                                      MPI_FLOAT, 
                                      queue->comm);
                    }
                    else{
                        MPI_Allgatherv(&layer->v_sumws[offset],
                                       length,
                                       MPI_FLOAT,
                                       layer->global_sumws,
                                       layer->scounts_weight,
                                       layer->sdispls_weight,
                                       MPI_FLOAT,
                                       queue->comm);
                    }
                }
            }
            pthread_mutex_lock(&queue->mut);
            queue->flag_gather_m1[layer_id] = 0;
            pthread_mutex_unlock(&queue->mut);
            pthread_cond_signal(&queue->cond);
        }
        else if(type == COMM_TYPE_FINISH){
            break;
        }
	}
	return (void *)NULL;
}
