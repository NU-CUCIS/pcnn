/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <string.h>
#include "def.h"
#include "model.h"
#include "util.h"

static void pcnn_pool_max_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i, j, k, l, m, n;
    int row, col;
    int offset;
    int maxrow, maxcol;
    int barea = bottom->output_rows * bottom->output_cols;
    int tarea = top->output_rows * top->output_cols;
    float max;
    float *activations = NULL;

    memset(top->e, 0, sizeof(float) * count * top->num_neurons);

#pragma omp parallel for private(j,k,l,m,n,row,col,max,maxrow,maxcol,offset,activations)
    for(i=0; i<top->output_depth; i++){
        for(j=0; j<count; j++){
            offset = (i*count+j)*tarea;
            activations = &bottom->a[(i*count+j)*barea];
            for(k=0; k<top->output_rows; k++){
                for(l=0; l<top->output_cols; l++){
                    max = -0xffff;
                    maxrow = 0;
                    maxcol = 0;
                    row = k*top->stride;
                    for(m=0; m<top->filter_rows; m++){
                        if(row < bottom->output_rows){
                            col = l*top->stride;
                            for(n=0; n<top->filter_cols; n++){
                                if(col < bottom->output_cols){
                                    if(max < activations[row*bottom->output_cols+col]){
                                        max = activations[row*bottom->output_cols+col];
                                        maxrow = row;
                                        maxcol = col;
                                    }
                                }
                                col++;
                            }
                        }
                        row++;
                    }
                    top->a[offset] = max;
                    top->poolmap[offset] = (i*count+j)*barea + maxrow*bottom->output_cols + maxcol;
                    offset++;
                }
            }
        }
    }
}

static void pcnn_pool_avg_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i, j, k, l, m, n;
    int r_start = 0;
    int c_start = 0;
    int r_end = 0;
    int c_end = 0;
    int pool_size = 0;
    float sum = 0;
    const int tarea = top->output_rows * top->output_cols;
    const int barea = bottom->output_rows * bottom->output_cols;
    float *input = NULL, *output = NULL;

    memset(top->a, 0, sizeof(float) * count * top->num_neurons);

#pragma omp parallel for private(j, k, l, m, n, pool_size, \
                                 input, output, sum, \
                                 r_start, r_end, \
                                 c_start, c_end)
    for(i=0; i<top->output_depth; i++){
        for(j=0; j<count; j++){
            input = &bottom->a[(i*count+j) * barea];
            output = &top->a[(i*count+j) * tarea];
            for(k=0; k<top->output_rows; k++){
                for(l=0; l<top->output_cols; l++){
                    sum = 0;
                    r_start = k * top->stride;
                    c_start = l * top->stride;
                    r_end = (r_start + top->filter_rows) < bottom->output_rows ? (r_start + top->filter_rows) : bottom->output_rows;
                    c_end = (c_start + top->filter_cols) < bottom->output_cols ? (c_start + top->filter_cols) : bottom->output_cols;
                    pool_size = (r_end - r_start) * (c_end - c_start);
                    for(m=r_start; m<r_end; m++){
                        for(n=c_start; n<c_end; n++){
                            sum += input[m * bottom->output_cols + n];
                        }
                    }
                    output[k*top->output_cols+l] += sum / pool_size;
                }
            }
        }
    }
}

void pcnn_pool_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i,j,k,l;
    int r_off, c_off, area;

    memset(top->a, 0, sizeof(float) * count * top->num_neurons);

    if(top->sub_type == 0)
        pcnn_pool_max_ff(op, count, bottom, top, param);
    else
        pcnn_pool_avg_ff(op, count, bottom, top, param);

    if(top->id == param->first_full_id-1){
        area = top->output_rows * top->output_cols;

#pragma omp parallel for private(j,k,l,r_off,c_off)
        for(i=0; i<top->output_depth; i++){
            r_off = i*count*area;
            c_off = i*count*area;
            for(j=0; j<top->output_rows; j++){
                for(k=0; k<top->output_cols; k++){
                    for(l=0; l<count; l++)
                        param->pool2full[r_off++] = top->a[c_off+(l*area)];;					
                    c_off++;
                }
            }
        }
    }
}

static void pcnn_pool_max_bp(int count, struct layer_t *bottom, struct layer_t *top)
{
    int i, j, k, l;
    int in_off = 0;
    int out_off = 0;

#pragma omp parallel for private(j,k,l,in_off,out_off)
    for(i=0; i<top->output_depth; i++){
        in_off = i * count * top->output_rows * top->output_cols;
        for(j=0; j<count; j++){
            for(k=0; k<top->output_rows; k++){
                for(l=0; l<top->output_cols; l++){
                    out_off = top->poolmap[in_off];
                    bottom->e[out_off] += top->e[in_off++];
                }
            }
        }
    }
}

static void pcnn_pool_avg_bp(int count, struct layer_t *bottom, struct layer_t *top)
{
    int i, j, k, l, m, n;
    int pool_size = 0;
    int r_start = 0;
    int c_start = 0;
    int r_end = 0;
    int c_end = 0;
    const int tarea = top->output_rows * top->output_cols;
    const int barea = bottom->output_rows * bottom->output_cols;
    float *input = NULL, *output = NULL;

    memset(bottom->e, 0, sizeof(float) * bottom->num_neurons * count);

#pragma omp parallel for private(j, k, l, m, n, pool_size, \
                                 input, output, \
                                 r_start, r_end, \
                                 c_start, c_end)
    for(i=0; i<top->output_depth; i++){
        for(j=0; j<count; j++){
            input = &top->e[(i*count+j) * tarea];
            output = &bottom->e[(i*count+j) * barea];
            for(k=0; k<top->output_rows; k++){
                for(l=0; l<top->output_cols; l++){
                    r_start = k * top->stride;
                    c_start = l * top->stride;
                    r_end = (r_start + top->filter_rows) < bottom->output_rows ? (r_start + top->filter_rows) : bottom->output_rows;
                    c_end = (c_start + top->filter_cols) < bottom->output_cols ? (c_start + top->filter_cols) : bottom->output_cols;
                    pool_size = (r_end - r_start) * (c_end - c_start);
                    for(m=r_start; m<r_end; m++){
                        for(n=c_start; n<c_end; n++){
                            output[m * bottom->output_cols + n] +=
                                input[k * top->output_cols + l] / pool_size;
                        }
                    }
                }
            }
        }
    }
}

void pcnn_pool_bp(int count, struct layer_t *bottom, struct layer_t *top)
{
    memset(bottom->e, 0, sizeof(float) * bottom->num_neurons * count);

    if(top->sub_type == 0)
        pcnn_pool_max_bp(count, bottom, top);
    else
        pcnn_pool_avg_bp(count, bottom, top);
}
