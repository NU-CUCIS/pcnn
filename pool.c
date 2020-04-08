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
    int i, j, k, l, m, n, o, p;
    int row, col, depth;
    int offset;
    int maxrow, maxcol, maxdepth;
    int barea = bottom->output_depth * bottom->output_rows * bottom->output_cols;
    int tarea = top->output_depth * top->output_rows * top->output_cols;
    float max;
    float *activations = NULL;

    memset(top->e, 0, sizeof(float) * count * top->num_neurons);

#pragma omp parallel for private(j,k,l,m,n,row,col,max,maxrow,maxcol,offset,activations)
    for(i=0; i<top->output_channels; i++){
        for(j=0; j<count; j++){
            offset = (i*count+j)*tarea;
            activations = &bottom->a[(i*count+j)*barea];
            for(o=0; o<top->output_depth; o++){
                for(k=0; k<top->output_rows; k++){
                    for(l=0; l<top->output_cols; l++){
                        max = -0xffff;
                        maxdepth = 0;
                        maxrow = 0;
                        maxcol = 0;
                        depth = o * top->stride_rows * top->stride_cols;
                        for(p=0; p<top->filter_depth; p++){
                            if(depth < bottom->output_depth){
                                row = k * top->stride_rows;
                                for(m=0; m<top->filter_rows; m++){
                                    if(row < bottom->output_rows){
                                        col = l*top->stride_cols;
                                        for(n=0; n<top->filter_cols; n++){
                                            if(col < bottom->output_cols){
                                                if(max < activations[depth * bottom->output_rows * bottom->output_cols + row * bottom->output_cols + col]){
                                                    max = activations[depth * bottom->output_rows * bottom->output_cols + row * bottom->output_cols + col];
                                                    maxdepth = depth;
                                                    maxrow = row;
                                                    maxcol = col;
                                                }
                                            }
                                            col++;
                                        }
                                    }
                                    row++;
                                }
                            }
                            depth++;
                        }
                        top->a[offset] = max;
                        top->poolmap[offset] = (i * count + j) * barea +
                                                maxdepth * bottom->output_rows * bottom->output_cols +
                                                maxrow * bottom->output_cols + maxcol;
                        offset++;
                    }//end of output_cols loop
                }// end of output_rows loop
            }// end of output_depth loop
        }//end of count loop
    }// end of output channels loop
}

static void pcnn_pool_avg_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i, j, k, l, m, n, o, p;
    int d_start = 0;
    int r_start = 0;
    int c_start = 0;
    int d_end = 0;
    int r_end = 0;
    int c_end = 0;
    int read_offset =0;
    int write_offset = 0;
    int pool_size = 0;
    float sum = 0;
    const int tarea = top->output_depth *
                      top->output_rows *
                      top->output_cols;
    const int barea = bottom->output_depth *
                      bottom->output_rows *
                      bottom->output_cols;
    float *input = NULL, *output = NULL;

    memset(top->a, 0, sizeof(float) * count * top->num_neurons);

#pragma omp parallel for private(j, k, l, m, n, o, p, \
                                 read_offset, write_offset, \
                                 sum, pool_size, \
                                 input, output, \
                                 d_start, d_end, \
                                 r_start, r_end, \
                                 c_start, c_end)
    for(i=0; i<top->output_channels; i++){
        for(j=0; j<count; j++){
            input = &bottom->a[(i*count+j) * barea];
            output = &top->a[(i*count+j) * tarea];
            write_offset = 0;
            for(k=0; k<top->output_depth; k++){
                for(l=0; l<top->output_rows; l++){
                    for(m=0; m<top->output_cols; m++){
                        sum = 0;
                        d_start = k * top->stride_depth;
                        r_start = l * top->stride_rows;
                        c_start = m * top->stride_cols;
                        d_end = (d_start + top->filter_depth) < bottom->output_depth ? (d_start + top->filter_depth) : bottom->output_depth;
                        r_end = (r_start + top->filter_rows) < bottom->output_rows ? (r_start + top->filter_rows) : bottom->output_rows;
                        c_end = (c_start + top->filter_cols) < bottom->output_cols ? (c_start + top->filter_cols) : bottom->output_cols;
                        pool_size = (d_end - d_start) * (r_end - r_start) * (c_end - c_start);
                        for(n=d_start; n<d_end; n++){
                            read_offset = n * bottom->output_rows * bottom->output_cols;
                            for(o=r_start; o<r_end; o++){
                                for(p=c_start; p<c_end; p++){
                                    sum += input[read_offset + (o * bottom->output_cols) + p];
                                }
                            }
                        }
                        output[write_offset++] += sum / pool_size;
                    }
                }
            }
        }
    }
}

void pcnn_pool_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param)
{
    int i,j,k,l,m;
    int r_off, c_off, area;

    memset(top->a, 0, sizeof(float) * count * top->num_neurons);

    if(top->sub_type == 0)
        pcnn_pool_max_ff(op, count, bottom, top, param);
    else
        pcnn_pool_avg_ff(op, count, bottom, top, param);

    if(top->id == param->first_full_id-1){
        area = top->output_depth * top->output_rows * top->output_cols;

#pragma omp parallel for private(j,k,l,m,r_off,c_off)
        for(i=0; i<top->output_channels; i++){
            r_off = i * count * area;
            c_off = i * count * area;
            for(j=0; j<top->output_depth; j++){
                for(k=0; k<top->output_rows; k++){
                    for(l=0; l<top->output_cols; l++){
                        for(m=0; m<count; m++)
                            param->pool2full[r_off++] = top->a[c_off + (m * area)];;					
                        c_off++;
                    }
                }
            }
        }
    }
}

static void pcnn_pool_max_bp(int count, struct layer_t *bottom, struct layer_t *top)
{
    int i, j, k, l, m;
    int in_off = 0;
    int out_off = 0;

#pragma omp parallel for private(j,k,l,m,in_off,out_off)
    for(i=0; i<top->output_channels; i++){
        in_off = i * count * top->output_rows * top->output_cols;
        for(j=0; j<count; j++){
            for(k=0; k<top->output_depth; k++){
                for(l=0; l<top->output_rows; l++){
                    for(m=0; m<top->output_cols; m++){
                        out_off = top->poolmap[in_off];
                        bottom->e[out_off] += top->e[in_off++];
                    }
                }
            }
        }
    }
}

static void pcnn_pool_avg_bp(int count, struct layer_t *bottom, struct layer_t *top)
{
    int i, j, k, l, m, n, o, p;
    int pool_size = 0;
    int d_start = 0;
    int r_start = 0;
    int c_start = 0;
    int d_end = 0;
    int r_end = 0;
    int c_end = 0;
    int read_offset = 0;
    int write_offset = 0;
    const int tarea = top->output_depth * top->output_rows * top->output_cols;
    const int barea = bottom->output_depth * bottom->output_rows * bottom->output_cols;
    float *input = NULL, *output = NULL;

    memset(bottom->e, 0, sizeof(float) * bottom->num_neurons * count);

#pragma omp parallel for private(j, k, l, m, n, o, p, \
                                 read_offset, write_offset, \
                                 pool_size, input, output, \
                                 d_start, r_start, c_start, \
                                 d_end, r_end, c_end)
    for(i=0; i<top->output_channels; i++){
        for(j=0; j<count; j++){
            input = &top->e[(i*count+j) * tarea];
            output = &bottom->e[(i*count+j) * barea];
            read_offset = 0;
            for(k=0; k<top->output_depth; k++){
                for(l=0; l<top->output_rows; l++){
                    for(m=0; m<top->output_cols; m++){
                        d_start = k * top->stride_depth;
                        r_start = l * top->stride_rows;
                        c_start = m * top->stride_cols;
                        d_end = (d_start + top->filter_depth) < bottom->output_depth ? (d_start + top->filter_depth) : bottom->output_depth;
                        r_end = (r_start + top->filter_rows) < bottom->output_rows ? (r_start + top->filter_rows) : bottom->output_rows;
                        c_end = (c_start + top->filter_cols) < bottom->output_cols ? (c_start + top->filter_cols) : bottom->output_cols;
                        pool_size = (d_end - d_start) * (r_end - r_start) * (c_end - c_start);
                        for(n=d_start; n<d_end; n++){
                            write_offset = n * bottom->output_rows * bottom->output_cols;
                            for(o=r_start; o<r_end; o++){
                                for(p=c_start; p<c_end; p++){
                                    output[write_offset + (o * bottom->output_cols) + p] += (input[read_offset] / pool_size);
                                }
                            }
                        }
                        read_offset++;
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
