/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include "def.h"
#include "config.h"
#include "model.h"
#include "frame.h"
#include "feeder.h"
#include "record.h"
#include "comm.h"
#include "arch.h"

int main(int argc, char **argv)
{
    int i;
    int opt=0;
    int ret=0;
    int mode=0;
    int num_epochs=1;
    int sync_interval=1;
    int do_shuffle=0;
    int continue_training=0;
    int num_groups=1;
    char *bin_path=NULL;
    extern char *optarg;
    struct model_t *model;
    struct param_t *param;
    struct comm_queue_t *queue;
    struct feeder_t *feeder;
    struct layer_t *layer;

    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &ret);
    if(ret != MPI_THREAD_MULTIPLE){
        printf("[%s][%d] MPI_Init_thread failed\n", __FUNCTION__,__LINE__);
    }

    while((opt=getopt(argc, argv, "s:e:t:m:g:i:"))!=EOF){
        if(opt == 'e'){
            num_epochs = atoi(optarg);
        }
        else if(opt == 's'){
            do_shuffle = atoi(optarg);
            do_shuffle = (do_shuffle>=1)?1:0;
        }
        else if(opt == 't'){
            continue_training = 1;
            bin_path = optarg;
        }
        else if(opt == 'm'){
            mode = atoi(optarg);
        }
        else if(opt == 'g'){
            num_groups = atoi(optarg);
        }
        else if(opt == 'i'){
            sync_interval = atoi(optarg);
        }
        else{
            printf("usage: %s -g [number of groups for averaged SGD] -e [number of epochs for training] -s [1: shuffle / 0: non-shuffle]\n", argv[0]);
            MPI_Finalize();
            return -1;
        }
    }

    /* Initialize communication module. */
    queue = pcnn_comm_init(num_groups, sync_interval);
    if(queue == NULL){
        printf("[%s][%d] init_queue failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    /* Initialize feeder module. */
    feeder = pcnn_feeder_init(do_shuffle, queue);
    if(feeder == NULL){
        printf("[%s][%d] pcnn_feeder_init failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    /* Initialize model module. */
    model = pcnn_model_init(feeder->num_train_images, num_epochs, mode, queue);
    if(model == NULL){
        printf("[%s][%d] init_model failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    /* Initialize a model as specified in init_model function. */
#if MNIST
    pcnn_arch_config_lenet(model, feeder);
#elif CIFAR10_MODEL
    pcnn_arch_config_cifar10(model, feeder);
#elif VGGA
    pcnn_arch_config_vgga(model, feeder);
#elif RESNET20
    pcnn_arch_config_resnet20(model, feeder);
#elif RESNET50
    pcnn_arch_config_resnet50(model, feeder);
#elif EDSR
    pcnn_arch_config_edsr(model, feeder);
#elif DRRN
    pcnn_arch_config_drrn(model, feeder);
#elif VDSR
    pcnn_arch_config_vdsr(model, feeder);
#endif

    /* Initialize parameter module */
    param = pcnn_model_init_param(model, feeder, queue);
    if(param == NULL){
        printf("[%s][%d] init_param failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    /* Initialize the communication data offsets. */
    pcnn_model_init_comm_offsets(model, queue);

    /* Read model parameters from the checkpoint file. */
    if(continue_training == 1)
        pcnn_record_continue_training(bin_path, param, model, queue);

    if(queue->group_id == 0 && queue->rank == 0){
        printf("---------------------------------------------------------\n");
        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];
            printf("Layer %-2d: %-4d x %-4d x %-4d x %-4d (ReLU: %d) (Batch Norm: %d)\n",
                    layer->id,layer->output_channels, layer->output_depth, layer->output_rows, layer->output_cols,
                    layer->ReLU, layer->batch_norm);
        }

        printf("---------------------------------------------------------\n");
        if(model->layers[model->num_layers-1]->loss_type == LOSS_TYPE_SOFTMAX){
            printf("%-40s: %s\n", "loss function", "softmax");
        }
        else if(model->layers[model->num_layers-1]->loss_type == LOSS_TYPE_MAE){
            printf("%-40s: %s\n", "loss function", "MAE (L1)");
        }
        else if(model->layers[model->num_layers-1]->loss_type == LOSS_TYPE_MSE){
            printf("%-40s: %s\n", "loss function", "MSE (L2)");
        }
        else{
            printf("loss function should be either softmax, MAE, or MSE!\n");
            return -1;
        }
        printf("---------------------------------------------------------\n");
        printf("%-40s: %d\n", "Number of communication groups", queue->num_groups);
        printf("%-40s: %d\n", "Synchronization interval", queue->sync_interval);
        printf("---------------------------------------------------------\n");
        printf("%-40s: %lu KBytes\n", "Memory space for parameters", model->param_size/1024);
        printf("%-40s: %lu KBytes\n", "Memory space for intermediate data", model->intermediate_size/1024);
        printf("---------------------------------------------------------\n");
        printf("%-40s: %d times\n", "Current number of updates", param->num_updates);
        printf("%-40s: %d times\n", "Current number of epochs", param->num_trained_epochs);
        printf("---------------------------------------------------------\n");
    }

    MPI_Barrier(queue->comm);

    /* Training begins. */
    pcnn_frame_run(feeder, model, param, queue);

    /* Release memory space for datasets */
    MPI_Barrier(queue->comm);
    pcnn_model_free_param(model, param);
    pcnn_model_destroy(model);
    pcnn_feeder_destroy(feeder);
    pcnn_comm_destroy(queue);

    MPI_Finalize();
    return 0;
}
