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
#include "transform.h"
#include "feeder.h"
#include "relu.h"

int pcnn_conv_ff(int op, int count, 
                 struct layer_t *bottom, struct layer_t *top, 
                 struct feeder_t *feeder, struct param_t *param)
{
    int i, j, k, l, m, area;
    int rows, cols;
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;

    rows = count * top->output_depth * top->output_rows * top->output_cols;
    cols = top->input_channels * top->filter_depth * top->filter_rows * top->filter_cols;

    /* 1. Im2col
    * Convert convolution operations to a single matrix multiplication. */
    if(bottom == NULL){
        pcnn_transform_im2col1(feeder->minibatch, param->col, count,
                               top->pad_depth, top->pad_rows, top->pad_cols,
                               top->stride_depth, top->stride_rows, top->stride_cols,
                               top->input_channels, top->input_depth, top->input_rows, top->input_cols,
                               top->output_depth, top->output_rows, top->output_cols,
                               top->filter_depth, top->filter_rows, top->filter_cols);
    }
    else{
        pcnn_transform_im2col2(bottom->a, param->col, count,
                               top->pad_depth, top->pad_rows, top->pad_cols,
                               top->stride_depth, top->stride_rows, top->stride_cols,
                               top->input_channels, top->input_depth, top->input_rows, top->input_cols,
                               top->output_depth, top->output_rows, top->output_cols,
                               top->filter_depth, top->filter_rows, top->filter_cols);
    }

    /* 2. GEMM
    * Use cblas or MKL to perform the matrix multiplication. */
    A = top->weight;
    B = param->col;
    C = top->a;
    M = top->output_channels;
    N = rows;
    K = cols;
    lda = K;
    ldb = N;
    ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);

    /* 3. Bias
    * Add the bias vector to the output activations. */
#pragma omp parallel for private(j)
    for(i=0; i<M; i++){
        for(j=0; j<N; j++)
            C[i*N+j] += top->bias[i];
    }

    /* Initialize errors in feed-forward stage. */
    memset(top->e, 0, sizeof(float) * count * top->num_neurons);

    /* If the next layer is the first fully-connected layer,
     * the data layout should be changed from OC x N x OD x OR x OC
     * to OC x ODx OR x OC x N. */
    if(top->id == param->first_full_id-1){
        area = top->output_depth * top->output_rows * top->output_cols;

#pragma omp parallel for private(j, k, l, m, rows, cols)
        for(i=0; i<top->output_channels; i++){
            rows = i * count * area;
            cols = i * count * area;
            for(j=0; j<top->output_depth; j++){
                for(k=0; k<top->output_rows; k++){
                    for(l=0; l<top->output_cols; l++){
                        for(m=0; m<count; m++)
                            param->pool2full[rows++] = top->a[rows + (m * area)];;					
                        cols++;
                    }
                }
            }
        }
    }
    return 0;
}

int pcnn_conv_bp(int op, int count,
                 struct layer_t *bottom, struct layer_t *top,
                 struct param_t *param)
{
    int M,K,N,lda,ldb,ldc;
    int rows, cols;
    float *A, *B, *C;

    if(bottom == NULL)
        return 0;

    rows = count * top->output_depth * top->output_rows * top->output_cols;
    cols = top->input_channels * top->filter_depth * top->filter_rows * top->filter_cols;

    /* 1. gemm */
    A = top->weight;
    B = top->e;
    C = param->col;
    M = cols;
    N = rows;
    K = top->output_channels;
    lda = M;
    ldb = N;
    ldc = N;

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);

    /* 2. col2im */
    pcnn_transform_col2im(bottom->e, param->col, count,
                          top->pad_depth, top->pad_rows, top->pad_cols, 
                          top->stride_depth, top->stride_rows, top->stride_cols, 
                          top->input_channels, top->input_depth, top->input_rows, top->input_cols, 
                          top->output_depth, top->output_rows, top->output_cols, 
                          top->filter_depth, top->filter_rows, top->filter_cols);

    return 0;
}

void pcnn_conv_gradw(int op, int count, 
                     struct layer_t *bottom, struct layer_t *top, 
                     struct feeder_t *feeder, struct param_t *param)
{
    int M,N,K,lda,ldb,ldc;
    float *A, *B, *C;

    if(bottom == NULL){
        pcnn_transform_im2col1(feeder->minibatch, param->col, count,
                               top->pad_depth, top->pad_rows, top->pad_cols,
                               top->stride_depth, top->stride_rows, top->stride_cols,
                               top->input_channels, top->input_depth, top->input_rows, top->input_cols,
                               top->output_depth, top->output_rows, top->output_cols,
                               top->filter_depth, top->filter_rows, top->filter_cols);
        A = top->e;
        B = param->col;
        C = top->local_sumws;
        M = top->output_channels;
        N = top->input_channels * top->filter_depth * top->filter_rows * top->filter_cols;
        K = count * top->output_depth * top->output_rows * top->output_cols;
        lda = K;
        ldb = K;
        ldc = N;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);
    }
    else{
        pcnn_transform_im2col_prime(bottom->a, param->col, count,
                                    top->pad_depth, top->pad_rows, top->pad_cols,
                                    top->stride_depth, top->stride_rows, top->stride_cols,
                                    top->input_channels, top->input_depth, top->input_rows, top->input_cols,
                                    top->output_depth, top->output_rows, top->output_cols,
                                    top->filter_depth, top->filter_rows, top->filter_cols);
        A = top->e;
        B = param->col;
        C = top->local_sumws;
        M = top->output_channels;
        N = top->input_channels * top->filter_depth * top->filter_rows * top->filter_cols;
        K = count * top->output_depth * top->output_rows * top->output_cols;
        lda = K;
        ldb = N;
        ldc = N;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);
    }
}

void pcnn_conv_gradb(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    float *A, *B, *C;
    int M, N, K, lda, ldb, ldc;

    A = layer->e;
    B = param->multiplier;
    C = layer->local_sumbs;
    M = layer->output_channels;
    N = 1;
    K = layer->output_depth * layer->output_rows * layer->output_cols * feeder->local_batch_size;
    lda = K;
    ldb = N;
    ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, 0., C, ldc);
}
