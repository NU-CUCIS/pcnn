/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include "def.h"
#include "model.h"
#include "comm.h"
#include "feeder.h"

int pcnn_record_checkpoint(struct param_t *param, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i;
    FILE *fd;
    char name[100];
    DIR *dir;
    struct layer_t *layer;

    dir = opendir(model->checkpoint_path);
    if(dir == NULL){
        printf("[%s][%d] opendir failed, %s doesn't exist!\n", __FUNCTION__, __LINE__, model->checkpoint_path);
        return -1;
    }

    /* When multi-step gradient averaging algorithm is used,
     * each worker updates its own part of the momentum parameters.
     * So, before rank0 performs checkpointing, gather the momentum parameters
     * from among all the processes. */
    if(queue->nproc > 1)
        pcnn_model_put_momentum_together(model, param, queue);

    if(queue->group_id == 0 && queue->rank == 0){
        if((param->num_trained_epochs % model->checkpoint_interval) == 0){
            sprintf(name, "%s/check-g%d-e%d.data", model->checkpoint_path, queue->group_id, param->num_trained_epochs);
            printf("checkpointing to %s num_updates: %d num_epochs: %d\n", name, param->num_updates, param->num_trained_epochs);

            fd = fopen(name, "w");
            /* Write the number of epochs first. */
            fwrite(&param->num_trained_epochs, sizeof(int) , 1, fd);

            /* Write the number of updates. */
            fwrite(&param->num_updates, sizeof(int) , 1, fd);

            /* Write the current learning rate. */
            fwrite(&model->learning_rate, sizeof(float) , 1, fd);

            /* Write the model parameter size */
            fwrite(&param->total_size, sizeof(int), 1, fd);

            /* Write the current model parameters. */
            fwrite(param->params, sizeof(float), param->total_size, fd);

            /* Write the current batch normalization scale factor. */
            fwrite(&param->bn_num_layers, sizeof(int), 1, fd);
            if(param->bn_num_layers > 0){
                for(i=0; i<model->num_layers; i++){
                    layer = model->layers[i];
                    if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1){
                        fwrite(&layer->bn_scale_factor, sizeof(float), 1, fd);
                        break;
                    }
                }
            }

            /* Write the global statistics for batch normalization if needed. */
            fwrite(&param->bn_global_statistics_size, sizeof(int), 1, fd);
            if(param->bn_global_statistics_size > 0)
                fwrite(param->bn_global_statistics, sizeof(float), param->bn_global_statistics_size, fd);

            /* Write the current batch normalization parameters if needed. */
            fwrite(&param->bn_param_size, sizeof(int), 1, fd);
            if(param->bn_param_size > 0)
                fwrite(param->bn_params, sizeof(float), param->bn_param_size, fd);

            /* Write the momentum vectors. */
            if(model->optimizer == OPTIMIZER_SGD){
                fwrite(param->prev_gradient_sums, sizeof(float), param->total_size, fd);
                if(param->bn_param_size > 0)
                    fwrite(param->bn_prev_gradients, sizeof(float), param->bn_param_size, fd);
            }
            else if(model->optimizer == OPTIMIZER_ADAM){
                fwrite(param->m_gradient_sums, sizeof(float), param->total_size, fd);
                fwrite(param->v_gradient_sums, sizeof(float), param->total_size, fd);
                /* beta decays */
                fwrite(&param->beta1_decay, sizeof(float) , 1, fd);
                fwrite(&param->beta2_decay, sizeof(float) , 1, fd);

                if(param->bn_param_size > 0){
                    fwrite(param->bn_m_gradients, sizeof(float), param->bn_param_size, fd);
                    fwrite(param->bn_v_gradients, sizeof(float), param->bn_param_size, fd);
                }
            }
            /* Write the previous lazy update interval. */
            fwrite(&param->interval, sizeof(int), 1, fd);

            fclose(fd);
        }
    }
    MPI_Barrier(queue->comm);
    return 0;
}

int pcnn_record_continue_training(char *bin_path, struct param_t *param, struct model_t *model, struct comm_queue_t *queue)
{
    int i;
    FILE *fd;
    int param_size;
    char file[100];
    char path[200];
    size_t ret;
    float bn_scale_factor;
    struct layer_t *layer;

    sprintf(file, "check-g%d.data", queue->group_id);
    sprintf(path, "%s/%s", bin_path, file);

    fd = fopen(path, "r");
    if(fd == NULL){
        printf("%s doesn't exist!\n", path);
        return -1;
    }

    ret = fread(&param->num_trained_epochs, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    ret = fread(&param->num_updates, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    ret = fread(&model->learning_rate, sizeof(float), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    ret = fread(&param_size, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size != param->total_size){
        printf("[%s][%d] model size is different to the recorded size!\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    ret = fread(param->params, sizeof(float), param->total_size, fd);
    if(ret != (size_t)param->total_size){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    ret = fread(&param_size, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size != param->bn_num_layers){
        printf("[%s][%d] number of layers with batch normalization is different from what was recorded before!\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param->bn_num_layers > 0){
        ret = fread(&bn_scale_factor, sizeof(float), 1, fd);
        if(ret != 1){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        for(i=0; i<model->num_layers; i++){
            layer = model->layers[i];
            if(layer->type == LAYER_TYPE_CONV && layer->batch_norm == 1)
                layer->bn_scale_factor = bn_scale_factor;
        }
    }

    ret = fread(&param_size, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size != param->bn_global_statistics_size){
        printf("[%s][%d] batch normalization global statistics parameter size is different to the recorded size!\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size > 0){
        ret = fread(param->bn_global_statistics, sizeof(float), param->bn_global_statistics_size, fd);
        if(ret != (size_t)param->bn_global_statistics_size){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }
    }

    ret = fread(&param_size, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size != param->bn_param_size){
        printf("[%s][%d] batch normalization parameter size is different to the recorded size!\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }

    if(param_size > 0){
        ret = fread(param->bn_params, sizeof(float), param->bn_param_size, fd);
        if(ret != (size_t)param->bn_param_size){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }
    }

    if(model->optimizer == OPTIMIZER_SGD){
        ret = fread(param->prev_gradient_sums, sizeof(float), param->total_size, fd);
        if(ret != (size_t)param->total_size){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        if(param->bn_param_size > 0){
            ret = fread(param->bn_prev_gradients, sizeof(float), param->bn_param_size, fd);
            if(ret != (size_t)param->bn_param_size){
                printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
                fclose(fd);
                return -1;
            }
        }
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        ret = fread(param->m_gradient_sums, sizeof(float), param->total_size, fd);
        if(ret != (size_t)param->total_size){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        ret = fread(param->v_gradient_sums, sizeof(float), param->total_size, fd);
        if(ret != (size_t)param->total_size){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        /* beta decays */
        ret = fread(&param->beta1_decay, sizeof(float) , 1, fd);
        if(ret != 1){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        ret = fread(&param->beta2_decay, sizeof(float) , 1, fd);
        if(ret != 1){
            printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
            fclose(fd);
            return -1;
        }

        if(param->bn_param_size > 0){
            ret = fread(param->bn_m_gradients, sizeof(float), param->bn_param_size, fd);
            if(ret != (size_t)param->bn_param_size){
                printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
                fclose(fd);
                return -1;
            }

            ret = fread(param->bn_v_gradients, sizeof(float), param->bn_param_size, fd);
            if(ret != (size_t)param->bn_param_size){
                printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
                fclose(fd);
                return -1;
            }
        }
    }

    ret = fread(&param->interval, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }
    
    fclose(fd);
    return 0;
}
