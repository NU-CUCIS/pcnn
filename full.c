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
#include "comm.h"
#include "util.h"
#include "feeder.h"
#include "relu.h"

int pcnn_full_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i,j;
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;

    /* matrix multiplication (W*A = A') */
    A = top->weight;
    B = bottom->type == LAYER_TYPE_POOL ? param->pool2full : bottom->a;
    C = top->a;
    M = top->num_neurons;
    N = count;
    K = top->num_prev_neurons;
    lda = M;
    ldb = N;
    ldc = N;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);

    /* bias addition */
#pragma omp parallel for private(j)
    for(i=0; i<M; i++){
        for(j=0; j<N; j++)
            C[i*N+j] += top->bias[i];
    }

    /* Initialize errors in feed-forward stage. */
    memset(top->e, 0, sizeof(float) * count * top->num_neurons);

    return 0;
}

int pcnn_full_bp(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i,j,k;
    int area, off;
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;
    int rowidx;

    /* 1. gemm */
    A = top->weight;
    B = top->e;
    C = bottom->type == LAYER_TYPE_FULL ? bottom->e : param->col;
    M = bottom->num_neurons;
    N = count;
    K = top->num_neurons;
    lda = K;
    ldb = N;
    ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);

    if(bottom->type != LAYER_TYPE_FULL){
        area = bottom->output_rows * bottom->output_cols;

#pragma omp parallel for private(j,k,off,rowidx)
        for(i=0; i<bottom->output_depth; i++){
            off = i*count*area;
            rowidx = i*area;
            for(j=0; j<count; j++){
                for(k=0; k<area; k++)
                    bottom->e[off++] = param->col[(rowidx+k)*count + j];
            }
        }
    }

    return 0;
}

void pcnn_full_gradw_pattern0(struct layer_t *bottom, struct layer_t *top, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;

    A = bottom->type == LAYER_TYPE_POOL ? param->pool2full : bottom->a; 
    B = top->e;
    C = top->local_sumws;

    M = bottom->num_neurons;
    N = top->num_neurons;
    K = feeder->local_batch_size;
    lda = K;
    ldb = K;
    ldc = N; 

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);
}

void pcnn_full_gradb_pattern0(struct layer_t *top, struct model_t *model, struct feeder_t *feeder)
{
    int i,j;
    float sum;

#pragma omp parallel for private(j,sum)
    for(i=0; i<top->num_neurons; i++){
        sum = 0;
        for(j=0; j<feeder->local_batch_size; j++)
            sum += top->e[i * feeder->local_batch_size + j];
        top->local_sumbs[i] = sum;
    }
}

void pcnn_full_gradw_pattern1(struct layer_t *bottom, struct layer_t *top, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;

    if(queue->nproc > 1)
        A = bottom->rep_a;
    else
        A = (top->id == param->first_full_id)?param->pool2full:bottom->a;
    B = queue->nproc > 1 ? top->rep_e : top->e;
    C = queue->nproc > 1 ? top->global_sumws : top->local_sumws;

    M = bottom->num_neurons / queue->nproc;
    N = top->num_neurons;
    K = feeder->local_batch_size * queue->nproc;
    lda = K;
    ldb = K;
    ldc = N; 

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);
}

void pcnn_full_gradb_pattern1(struct layer_t *top, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i,j;
    float sum;
    float *error;
    float *sumbs;

    error = (queue->nproc>1)?top->rep_e:top->e;
    sumbs = (queue->nproc>1)?top->global_sumbs:top->local_sumbs;

#pragma omp parallel for private(j,sum)
    for(i=0; i<top->num_neurons; i++){
        sum = 0;
        for(j=0; j<feeder->batch_size; j++)
            sum += error[i * feeder->batch_size + j];
        sumbs[i] = sum;
    }
}
