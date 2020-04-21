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
#include "model.h"
#include "conv.h"
#include "full.h"
#include "pool.h"
#include "util.h"
#include "feeder.h"
#include "comm.h"
#include "record.h"
#include "relu.h"
#include "loss.h"
#include "batch_norm.h"
#include "residual.h"
#include "ffbp_allreduce.h"
#include "ffbp_multistep.h"

/* static function declarations */
static void pcnn_frame_process_batch(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
static void pcnn_frame_test(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
static int pcnn_frame_train(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
static void *pcnn_frame_thread(void *ptr);

static void pcnn_frame_process_batch(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    /* 1. Feed-forward stage. */
    (*model->feedforward)(OPERATION_TYPE_TRAINING, feeder, model, param, queue);

    /* 2. Backpropagation stage. */
    (*model->backprop)(OPERATION_TYPE_TRAINING, feeder, model, param, queue);

    /* 3. Update the model parameters. */
    (*model->update)(model, param, feeder, queue);
}

static void pcnn_frame_test(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    int acc=0;
    float PSNR;
    FILE *fd=NULL;
    struct comm_req_t req;

    param->num_corrects = 0;
    param->local_loss = 0;
    param->num_processed_batches = 0;
    param->current_test_index = 0;

    /* Feed-forward all the test samples. */
    while((param->current_test_index + feeder->batch_size) < feeder->num_test_images){
        pcnn_feeder_get_minibatch(1, param->current_test_index, model, feeder, queue);
        pcnn_feeder_subtract_mean_image(feeder);
        (*model->feedforward)(OPERATION_TYPE_VALIDATION, feeder, model, param, queue);
        param->current_test_index += (feeder->batch_size * queue->num_groups);
        param->num_processed_batches++;
        
#if DEBUG
        if(model->task_type == TASK_TYPE_CLASSIFICATION){
            printf("R%d [%s][%d] (%d/%d) Accurate predictions on the test samples: %d/%d\n",
                    queue->rank, __FUNCTION__,__LINE__, param->current_test_index,
                                           feeder->num_test_images,
                                           param->num_corrects,
                                           param->current_test_index);
        }
        else if(model->task_type == TASK_TYPE_REGRESSION){
            printf("[%d/%d] testing... the average PSNR: %f\n", param->current_test_index,
                                                                feeder->num_test_images,
                                                                param->local_loss / (float)param->num_processed_batches);
        }
#endif
    }
	
    /* Reduce the local number of accurate predictions across all the workers. */
    if(model->task_type == TASK_TYPE_CLASSIFICATION){
        if(queue->nproc * queue->num_groups > 1)
            MPI_Allreduce(&param->num_corrects, &acc, 1, MPI_INT, MPI_SUM, queue->world);
        else
            acc = param->num_corrects;

        if(queue->group_id == 0 && queue->rank == 0){
            fd = fopen("acc.txt", "a+");
            fprintf(fd, "%11.8f\n", (float)acc * (float)100 / (float)param->current_test_index);
            fflush(fd);
            fclose(fd);
        }
        printf("Validation accuracy: %4.2f\n", (float)acc * (float)100 / (float)param->current_test_index);
    }
    else if(model->task_type == TASK_TYPE_REGRESSION){
        if(queue->num_groups * queue->nproc > 1){
            req.type = COMM_TYPE_REDUCE_L;
            pcnn_comm_insert_req(model, queue, &req);

            pthread_mutex_lock(&queue->mut);
            while(queue->flag_reduce_l == 1)
                pthread_cond_wait(&queue->cond, &queue->mut);
            pthread_mutex_unlock(&queue->mut);
            param->epoch_loss = param->global_loss / (queue->nproc * queue->num_groups);
        }
        else{
            param->epoch_loss = param->local_loss;
        }
        param->epoch_loss /= param->num_processed_batches;

        if(queue->group_id == 0 && queue->rank == 0){
            fd = fopen("acc.txt", "a+");
            fprintf(fd, "%11.8f\n", param->epoch_loss);
            fflush(fd);
            fclose(fd);
            printf("Validation loss: %4.2f\n", param->epoch_loss);
        }
    }

    pcnn_record_checkpoint(param, model, feeder, queue);
}

static int pcnn_frame_train(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    int i;
    double time;
    struct comm_req_t req;
    
    if(model->mode == 1){
        printf("Evaluation Mode\n");
        pcnn_frame_test(feeder, model, param, queue);
        return 0;
    }

    for(i=0; i<model->num_epochs; i++){
        param->epoch = i;
        param->num_corrects = 0;
        param->custom_output = 0.0f;
        param->local_loss = 0;
        param->num_processed_batches = 0;

        if(queue->group_id == 0 && queue->rank == 0)
            printf("Epoch %d/%d lr: %f batch size: %d\n", param->epoch + 1, model->num_epochs, model->learning_rate, feeder->batch_size); 

        if(feeder->do_shuffle)
            pcnn_feeder_shuffle(feeder, queue);

        /* Decay the learning rate if needed. */
        pcnn_model_decay_learning_rate(model, param);

        /* Generate the random offsets for cropping the training images. */
        pcnn_feeder_generate_crop_offsets(feeder);

        time = MPI_Wtime();

        param->current_index = 0;
        while((param->current_index + (feeder->batch_size * queue->num_groups)) <= feeder->num_train_images){
            if(model->optimizer == OPTIMIZER_ADAM){
                param->beta1_decay *= model->beta1;
                param->beta2_decay *= model->beta2;
            }

            pcnn_feeder_get_minibatch(0, param->current_index, model, feeder, queue);
            pcnn_feeder_subtract_mean_image(feeder);
            pcnn_frame_process_batch(feeder, model, param, queue);
            param->current_index += (feeder->batch_size * queue->num_groups);
            param->num_processed_batches++;

            if(param->num_processed_batches % 10 == 0){
                if(model->task_type == TASK_TYPE_CLASSIFICATION)
                    printf("[%d/%d] training... number of accurate predictions: %d\n", param->current_index,
                                                                                       feeder->num_train_images,
                                                                                       param->num_corrects);
                else if(model->task_type == TASK_TYPE_REGRESSION)
                    printf("[%d/%d] training... the training loss: %f\n", param->current_index,
                                                                          feeder->num_train_images,
                                                                          param->local_loss / (float)param->num_processed_batches);
            }

            if(model->test_per_epoch == 0 &&
               param->num_updates % 1000 == 0 &&
               param->num_updates != 0){
                    pcnn_frame_test(feeder, model, param, queue);
            }
        }
        param->num_trained_epochs++;

        /* Reduce the loss across all the processe.s */
        if(queue->nproc > 1){
            req.type = COMM_TYPE_REDUCE_L;
            pcnn_comm_insert_req(model, queue, &req);

            pthread_mutex_lock(&queue->mut);
            while(queue->flag_reduce_l == 1)
                pthread_cond_wait(&queue->cond, &queue->mut);
            pthread_mutex_unlock(&queue->mut);
            param->epoch_loss = (param->global_loss / (queue->nproc  * queue->num_groups));
        }
        else{
            param->epoch_loss = param->local_loss;
        }
        param->epoch_loss /= (float)param->num_processed_batches;

        if(model->task_type == TASK_TYPE_CLASSIFICATION){
            printf("[%d/%d] training... number of accurate predictions: %d\n", param->current_index,
                                                                               feeder->num_train_images,
                                                                               param->num_corrects);
        }
        else if(model->task_type == TASK_TYPE_REGRESSION){
            printf("[%d/%d] training... the training loss: %f\n", param->current_index,
                                                                  feeder->num_train_images,
                                                                  param->epoch_loss);
        }
        printf("%d batches has been processed by each process in %f sec, epoch_loss: %f\n", param->num_processed_batches,
                                                                                            MPI_Wtime() - time,
                                                                                            param->epoch_loss);
        if(queue->rank == 0){
            FILE *fd;
            char name[100];
            sprintf(name, "loss-g%d.txt", queue->group_id);
            fd = fopen(name, "a");
            fprintf(fd, "%f\n", param->epoch_loss);
            fclose(fd);
        }

        if(model->test_per_epoch != 0)
            pcnn_frame_test(feeder, model, param, queue);
    }
    return 0;
}

static void *pcnn_frame_thread(void *ptr)
{
    int ret;
    struct msg_t *msg;
    struct model_t *model;
    struct param_t *param;
    struct feeder_t *feeder;
    struct comm_req_t req;
    struct comm_queue_t *queue;

    msg = (struct msg_t *)ptr;
    model = (struct model_t *)msg->model;
    param = (struct param_t *)msg->param;
    feeder = (struct feeder_t *)msg->feeder;
    queue = (struct comm_queue_t *)msg->queue;

    ret = pcnn_frame_train(feeder, model, param, queue);
    if(ret != 0){
        printf("[%s][%d] pcnn_frame_train failed\n", __FUNCTION__,__LINE__);
        return (void *)NULL;
    }

    if(queue->nproc * queue->num_groups > 1){
        req.type = COMM_TYPE_FINISH;
        pcnn_comm_insert_req(model, queue, &req);
    }
    return (void *)NULL;
}

int pcnn_frame_run(struct feeder_t *feeder, struct model_t *model, struct param_t *param, struct comm_queue_t *queue)
{
    struct msg_t msg;

    /* Plug-in the proper functions into the model module. */
    if(model->comm_pattern == 0){
        model->feedforward = &pcnn_ffbp_allreduce_feedforward;
        model->backprop = &pcnn_ffbp_allreduce_backprop;
        model->update = &pcnn_ffbp_allreduce_update;
    }
    else if(model->comm_pattern == 1){
        model->feedforward = &pcnn_ffbp_multistep_feedforward;
        model->backprop = &pcnn_ffbp_multistep_backprop;
        model->update = &pcnn_ffbp_multistep_update;
    }
    else{
        printf("[%s][%d] communicaiton pattern is not set correctly.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    /* Create compute thread and communication thread. */
    msg.model = model;
    msg.param = param;
    msg.feeder = feeder;
    msg.queue = queue;

    pthread_create(&model->comp_thread, NULL, pcnn_frame_thread, (void *)&msg);
    if(queue->nproc * queue->num_groups > 1)
        pthread_create(&model->comm_thread, NULL, pcnn_comm_thread, (void *)&msg);

    /* Join the threads. */
    pthread_join(model->comp_thread, NULL);
    if(queue->nproc * queue->num_groups > 1)
        pthread_join(model->comm_thread, NULL);

    return 0;
}
