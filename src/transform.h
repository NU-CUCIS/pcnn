/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_transform_im2col1(float *data, float *column_buffer, int count,
                            int pad_depth, int pad_rows, int pad_cols,
                            int stride_depth, int stride_rows, int stride_cols,
                            int input_channels, int input_depth, int input_rows, int input_cols,
                            int output_depth, int output_rows, int output_cols,
                            int filter_depth, int filter_rows, int filter_cols);

void pcnn_transform_im2col2(float *data, float *column_buffer, int count,
                            int pad_depth, int pad_rows, int pad_cols,
                            int stride_depth, int stride_rows, int stride_cols,
                            int input_channels, int input_depth, int input_rows, int input_cols,
                            int output_depth, int output_rows, int output_cols,
                            int filter_depth, int filter_rows, int filter_cols);

void pcnn_transform_im2col_prime(float *data, float *column_buffer, int count,
                                 int pad_depth, int pad_rows, int pad_cols,
                                 int stride_depth, int stride_rows, int stride_cols,
                                 int input_channels, int input_depth, int input_rows, int input_cols,
                                 int output_depth, int output_rows, int output_cols,
                                 int filter_depth, int filter_rows, int filter_cols);

void pcnn_transform_col2im(float *data, float *column_buffer, int count,
                           int pad_depth, int pad_rows, int pad_cols,
                           int stride_depth, int stride_rows, int stride_cols,
                           int input_channels, int input_depth, int input_rows, int input_cols,
                           int output_depth, int output_rows, int output_cols,
                           int filter_depth, int filter_rows, int filter_cols);

int pcnn_transform_rearrange(float *src, float *dest, int rows, int cols, int num_procs);
