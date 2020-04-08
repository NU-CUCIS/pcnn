/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "def.h"
#include "model.h"

/* Subpixel layer proposed in the following paper.
 * subpixel: A subpixel convolutional neural network implementation with Tensorflow
 * This layer is used for super-resolution model such as EDSR. 
 * It is assumed that the bottom layer is a convolution layer. */
void pcnn_upsample_ff(int count, int ratio, struct layer_t *bottom, struct layer_t *top)
{
    /* In the feed-forward stage, the input(bottom) layout is r^2D x B x H x W
     * and the output(top) layout is D x B x rH x rW. */
    int i, j, k, l;
    int src_off;
    int dst_off;
    int src_area;
    int dst_area;
    int src_row, src_col, src_depth;
    
    src_area = bottom->output_rows * bottom->output_cols;
    dst_area = count * top->output_rows * top->output_cols;

#pragma omp parallel for private(j, k, l, src_row, src_col, src_depth, src_off, dst_off)
    for (i = 0; i < top->output_channels; i++) {
        dst_off = i * dst_area; 
        for (j = 0; j < count; j++) {
            for(k = 0; k < top->output_rows; k++) {
                src_row = (int)(floor(k / ratio));
                for(l = 0; l < top->output_cols; l++) {
                    src_col = (int)(floor(l / ratio));
                    src_depth = top->output_channels * ratio * (l % ratio) + top->output_channels * (k % ratio) + i;
                    src_off = (src_depth * count * src_area) + (j * src_area) + (src_row * bottom->output_rows) + src_col;
                    top->a[dst_off++] = bottom->a[src_off];
                }
            }
        }
    }
}

void pcnn_upsample_bp(int count, int ratio, struct layer_t *bottom, struct layer_t *top)
{
    /* In the backpropagation stage, the input(top) layout is D x B x rH x rW
     * and the output(bottom) layout is r^2D x B x H x W. */
    int i, j, k, l;
    int src_off;
    int dst_off;
    int src_area;
    int dst_area;
    int dst_row, dst_col, dst_depth;
    
    dst_area = bottom->output_rows * bottom->output_cols;
    src_area = count * top->output_rows * top->output_cols;

#pragma omp parallel for private(j, k, l, dst_row, dst_col, dst_depth, src_off, dst_off)
    for (i = 0; i < top->output_channels; i++) {
        src_off = i * src_area; 
        for (j = 0; j < count; j++) {
            for(k = 0; k < top->output_rows; k++) {
                dst_row = (int)(floor(k / ratio));
                for(l = 0; l < top->output_cols; l++) {
                    dst_col = (int)(floor(l / ratio));
                    dst_depth = top->output_channels * ratio * (l % ratio) + top->output_channels * (k % ratio) + i;
                    dst_off = (dst_depth * count * dst_area) + (j * dst_area) + (dst_row * bottom->output_rows) + dst_col;
                    bottom->e[dst_off] = top->e[src_off++];
                }
            }
        }
    }
}
