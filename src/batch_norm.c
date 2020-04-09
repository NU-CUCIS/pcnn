/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_MKL 
#include <mkl.h>
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include "def.h"
#include "model.h"
#include "feeder.h"
#include "comm.h"

static void pcnn_bn_normalize_ff(int op, struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_bn_scale_shift_ff(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_bn_normalize_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
static void pcnn_bn_scale_shift_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);

void pcnn_bn_ff(int op, struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    /* Normalize the activations first. */
    pcnn_bn_normalize_ff(op, layer, model, param, feeder);

    /* Then, scale and shift. */
    pcnn_bn_scale_shift_ff(layer, model, param, feeder);
}

void pcnn_bn_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    /* Scale and shift. */
    pcnn_bn_scale_shift_bp(layer, model, param, feeder);

    /* Then, compute the gradients of gamma and beta. */
    pcnn_bn_normalize_bp(layer, model, param, feeder);
}

static void pcnn_bn_normalize_ff(int op, struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i;
    float scale = 0.0f;
    float variance_correction_factor;
    float *mean = NULL;
    float *variance = NULL;
    float *a, *b, *c, *x, *y;
    float alpha, beta;
    float bn_scale_factor;
    int lda, ldb, ldc;
    int m, n, k;
    int incx, incy;
    int channel_size;

    mean = (float *)malloc(sizeof(float) * layer->output_channels);
    variance = (float *)malloc(sizeof(float) * layer->output_channels);

    channel_size = layer->output_depth *
                   layer->output_rows *
                   layer->output_cols;
    scale = 1.0f / (feeder->local_batch_size * channel_size);

    /* Calculate the mean outputs. */
    if(op == OPERATION_TYPE_TRAINING){
        a = layer->a;
        x = param->multiplier;
        y = mean;
        m = layer->output_channels;
        n = feeder->local_batch_size * channel_size;
        alpha = scale;
        beta = 0.0f;
        lda = n;
        incx = 1;
        incy = 1;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
    else{
        bn_scale_factor = (float)1.0f / layer->bn_scale_factor;
#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++){
            mean[i] = layer->global_mean[i] * bn_scale_factor;
        }
    }

    /* Subtract the mean from output. */
    a = mean;
    b = param->multiplier;
    c = param->sums;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    
    a = param->sums;
    b = param->multiplier;
    c = layer->a;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    k = 1;
    alpha = -1.0f;
    beta = 1.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    if(op == OPERATION_TYPE_TRAINING){
        /* Calculate the variance of outputs. */
#pragma omp parallel for 
        for(i=0; i<feeder->local_batch_size * layer->num_neurons; i++)
            layer->sqrt_var[i] = powf(layer->a[i], 2);

        a = layer->sqrt_var;
        x = param->multiplier;
        y = variance;
        m = layer->output_channels;
        n = feeder->local_batch_size * channel_size;
        alpha = scale;
        beta = 0.0f;
        lda = n;
        incx = 1;
        incy = 1;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

        /* Calculate the moving average.
         * This will be used in testing. */
        layer->bn_scale_factor *= model->moving_average_fraction;
        layer->bn_scale_factor += 1;

        n = layer->output_channels;
        alpha = 1.0f;
        beta = model->moving_average_fraction;
        x = mean;
        y = layer->global_mean;
        incx = 1;
        incy = 1;
        cblas_saxpby(n, alpha, x, incx, beta, y, incy);

        m = feeder->local_batch_size * channel_size; 
        variance_correction_factor = (m > 1) ? (float)(m)/(m-1) : 1;

        n = layer->output_channels;
        alpha = variance_correction_factor;
        beta = model->moving_average_fraction;
        x = variance;
        y = layer->global_variance;
        incx = 1;
        incy = 1;
        cblas_saxpby(n, alpha, x, incx, beta, y, incy);
    }
    else{
        bn_scale_factor = (float)1.0f / layer->bn_scale_factor;
#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++){
            variance[i] = layer->global_variance[i] * bn_scale_factor;
        }
    }

#pragma omp parallel for
    for(i=0; i<layer->output_channels; i++){
        variance[i] += model->eps;
        variance[i] = sqrtf(variance[i]);
    }

    a = variance;
    b = param->multiplier;
    c = param->sums;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    
    a = param->sums;
    b = param->multiplier;
    c = layer->sqrt_var;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    /* Normalize the output. */
#pragma omp parallel for
    for(i=0; i<feeder->local_batch_size * layer->num_neurons; i++)
        layer->a[i] /= layer->sqrt_var[i];

    /* Keep the outputs before it goes through the activation function.
     * These will be used in backpropagation to calculate the gradients. */
    memcpy(layer->a_norm, layer->a, sizeof(float) * feeder->local_batch_size * layer->num_neurons);

    /* Free the intermediate memory spaces. */
    free(mean);
    free(variance);
}

static void pcnn_bn_scale_shift_ff(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i, j;
    int area;
    float beta;

    area = feeder->local_batch_size *
           layer->output_depth *
           layer->output_rows *
           layer->output_cols;

    /* Multiply gamma first (scaling). */
    for (i = 0; i < layer->output_channels; i++)
        cblas_sscal(area, layer->gamma[i], &layer->a[i * area], 1);

    /* Then, add beta (shifting). */
#pragma omp parallel for private(j, beta)
    for (i = 0; i < layer->output_channels; i++) {
        beta = layer->beta[i];
        for (j = 0; j < area; j++) {
            layer->a[i * area + j] += beta;
        }
    }
}

static void pcnn_bn_normalize_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i = 0;
    float *temp = NULL;
    float *mean = NULL;
    float *a, *b, *c, *x, *y;
    float alpha, beta, scale;
    int lda, ldb, ldc;
    int m, n, k;
    int incx, incy;
    int channel_size;

    channel_size = layer->output_depth *
                   layer->output_rows *
                   layer->output_cols;
    scale = (float)-1. / (float)(feeder->local_batch_size * channel_size);

    temp = param->col;
    mean = &param->col[feeder->local_batch_size * layer->num_neurons];

    memcpy(temp, layer->e, sizeof(float) * feeder->local_batch_size * layer->num_neurons);

#pragma omp parallel for
    for(i=0; i<feeder->local_batch_size * layer->num_neurons; i++)
        layer->e[i] = temp[i] * layer->a_norm[i];

    /* Calculate the mean outputs. */
    a = layer->e;
    x = param->multiplier;
    y = param->sums;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    a = param->sums;
    x = param->multiplier;
    y = mean;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    /* Replicate the mean. */
    a = mean;
    b = param->multiplier;
    c = param->sums;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    
    a = param->sums;
    b = param->multiplier;
    c = layer->e;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

#pragma omp parallel for
    for(i=0; i<feeder->local_batch_size * layer->num_neurons; i++)
        layer->e[i] = layer->e[i] * layer->a_norm[i];

    a = temp;
    x = param->multiplier;
    y = param->sums;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    a = param->sums;
    x = param->multiplier;
    y = mean;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    /* Replicate the mean. */
    a = mean;
    b = param->multiplier;
    c = param->sums;
    m = layer->output_channels;
    n = feeder->local_batch_size;
    k = 1;
    alpha = 1.0f;
    beta = 0.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    
    a = param->sums;
    b = param->multiplier;
    c = layer->e;
    m = feeder->local_batch_size * layer->output_channels;
    n = channel_size;
    k = 1;
    alpha = 1.0f;
    beta = 1.0f;
    lda = k;
    ldb = n;
    ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    
    n = feeder->local_batch_size * layer->num_neurons;
    alpha = 1.0f;
    beta = scale;
    x = temp;
    y = layer->e;
    incx = 1;
    incy = 1;
    cblas_saxpby(n, alpha, x, incx, beta, y, incy);

#pragma omp parallel for
    for(i=0; i<feeder->local_batch_size * layer->num_neurons; i++)
        layer->e[i] = layer->e[i] / layer->sqrt_var[i];
}

static void pcnn_bn_scale_shift_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder)
{
    int i;
    int area;
    int m, n, lda, incx, incy;
    float *a;
    float *x;
    float *y;
    float alpha, beta;

    area = feeder->local_batch_size *
           layer->output_depth *
           layer->output_rows *
           layer->output_cols;

    /* 1. compute beta gradients */
    a = layer->e;
    x = param->multiplier;
    y = layer->local_dbeta;
    m = layer->output_channels;
    n = area;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    /* 2. compute gamma gradients */
#pragma omp parallel for
    for (i = 0; i < layer->output_channels * area; i++) {
        param->col[i] = layer->e[i] * layer->a_norm[i];
    }

    a = param->col;
    x = param->multiplier;
    y = layer->local_dgamma;
    m = layer->output_channels;
    n = area;
    alpha = 1.0f;
    beta = 0.0f;
    lda = n;
    incx = 1;
    incy = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);

    /* 3. error backward propagation */
    for (i = 0; i < layer->output_channels; i++)
        cblas_sscal(area, layer->gamma[i], &layer->e[i * area], 1);
}

void pcnn_bn_update(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i;
    float correction;
    float *weight_gradient_sums;
    float *bias_gradient_sums;
    const float scale = 1.0f / feeder->local_batch_size;

    /* Only convolution layer is compatible with batch normalization. */
    if(layer->type != LAYER_TYPE_CONV)
        return;

    if(queue->nproc > 1){
        /* Curretly, our distributed batch normalization is not synchronous.
         * Assuming the local batch size and number of neurons at each layer are 
         * large enough, each process normalizes the activations locally.
         * It is not well-studied how much it affects the convergence, but
         * empirically it doesn't much affect the quality of results. */
        weight_gradient_sums = layer->local_dgamma;
        bias_gradient_sums = layer->local_dbeta;
    }
    else{
        weight_gradient_sums = layer->local_dgamma;
        bias_gradient_sums = layer->local_dbeta;
    }

    if(model->optimizer == OPTIMIZER_SGD){
        /* Note that we do not apply the weight decay method on
         * the batch normalization coefficients. These batch normalization
         * coefficients are locally trained using a small batch size.
         * Regularization is not necessary in this case. */
        cblas_sscal(layer->output_channels, scale, weight_gradient_sums, 1);
        cblas_saxpby(layer->output_channels, model->learning_rate, weight_gradient_sums, 1, model->momentum, layer->prev_dgamma, 1);
        cblas_saxpy(layer->output_channels, -1.0f, layer->prev_dgamma, 1, layer->gamma, 1);

        cblas_sscal(layer->output_channels, scale, bias_gradient_sums, 1);
        cblas_saxpby(layer->output_channels, model->learning_rate, bias_gradient_sums, 1, model->momentum, layer->prev_dbeta, 1);
        cblas_saxpy(layer->output_channels, -1.0f, layer->prev_dbeta, 1, layer->beta, 1);
    }
    else if(model->optimizer == OPTIMIZER_ADAM){
        cblas_saxpby(layer->output_channels, (1.f - model->beta1) * scale, weight_gradient_sums, 1, model->beta1, layer->m_dgamma, 1);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            weight_gradient_sums[i] = powf(weight_gradient_sums[i], 2);
        cblas_saxpby(layer->output_channels, 1.f - model->beta2, weight_gradient_sums, 1, model->beta2, layer->v_dgamma, 1);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_dgamma[i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            param->col[i] = param->col[i] * layer->m_dgamma[i];

        correction = sqrtf(1.f - param->beta2_decay) / (1.f - param->beta1_decay);
        cblas_saxpby(layer->output_channels, -1.f * model->learning_rate * correction, param->col, 1, 1.f, layer->gamma, 1); 

        cblas_saxpby(layer->output_channels, (1.f - model->beta1) * scale, bias_gradient_sums, 1, model->beta1, layer->m_dbeta, 1);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            bias_gradient_sums[i] = powf(bias_gradient_sums[i], 2);
        cblas_saxpby(layer->output_channels, 1.f - model->beta2, bias_gradient_sums, 1, model->beta2, layer->v_dbeta, 1);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            param->col[i] = 1.f / (sqrtf(layer->v_dbeta[i]) + model->epsilon);

#pragma omp parallel for
        for(i=0; i<layer->output_channels; i++)
            param->col[i] = param->col[i] * layer->m_dbeta[i];

        cblas_saxpby(layer->output_channels, -1.f * model->learning_rate * correction, param->col, 1, 1.f, layer->beta, 1); 
    }

    /* Initialize memory space for the next iteration. */
    memset(layer->local_dgamma, 0, sizeof(float) * layer->output_channels);
    memset(layer->local_dbeta, 0, sizeof(float) * layer->output_channels);
}
