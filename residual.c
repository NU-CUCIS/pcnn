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
#include "comm.h"
#include "feeder.h"

void pcnn_residual_ff(struct layer_t *right, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i,j,k,l;
    int src_index = 0;
    int dst_index = 0;
    struct layer_t *left = NULL;

    if (right->skip_from >= 0) {
        left = model->layers[right->skip_from];
#pragma omp parallel for
        for(i=0; i<feeder->local_batch_size * right->num_neurons; i++)
            right->a[i] *= right->res_scale;
	    cblas_saxpy(feeder->local_batch_size * right->num_neurons, 1.f, left->a, 1, right->a, 1);
    }
    else if (right->skip_from == -1) {
        /* Add the input images into the activations directly.
         * The images are stored as BDHW order while the convolution layer activations
         * are stored as DBHW. */
#pragma omp parallel for private(j, k, l, src_index, dst_index)
        for (i = 0; i < right->output_channels; i++) {
            dst_index = i * feeder->local_batch_size * right->output_rows * right->output_cols;
            for (j = 0; j < feeder->local_batch_size; j++) {
                src_index = (queue->rank * feeder->local_batch_size + j) * right->output_channels + i;
                src_index *= (right->output_rows * right->output_cols);
                for (k = 0; k < right->output_rows; k++) {
                    for (l = 0; l < right->output_cols; l++) {
                        right->a[dst_index++] += feeder->minibatch[src_index++];
                    }
                }
            }
        }
    }
}

void pcnn_residual_bp(struct layer_t *right, struct model_t *model, struct feeder_t *feeder)
{
    int i;
    struct layer_t *left = NULL;

    if(right->skip_from >= 0){
        left = model->layers[right->skip_from];
	    cblas_saxpy(feeder->local_batch_size * right->num_neurons, 1.f, right->e, 1, left->e, 1);
#pragma omp parallel for
        for(i=0; i<feeder->local_batch_size * right->num_neurons; i++)
            right->e[i] *= right->res_scale;
    }
}
