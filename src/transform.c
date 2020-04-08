/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* This function is brought from Caffe source code (caffe/util/im2col.cpp) */
inline int is_a_ge_zero_and_a_lt_b(int a, int b){
	return (unsigned)(a) < (unsigned)(b);
}

/* Input: (N) x (IF x ID x IR x IC)
 * Output: (IF x FD x FR x FC) x (N x OD x OR x OC) */
void pcnn_transform_im2col1(float *data, float *column_buffer, int count,
                            int pad_depth, int pad_rows, int pad_cols,
                            int stride_depth, int stride_rows, int stride_cols,
                            int input_channels, int input_depth, int input_rows, int input_cols,
                            int output_depth, int output_rows, int output_cols,
                            int filter_depth, int filter_rows, int filter_cols)
{
    size_t i,j,k,l,m,n,o,p;
    size_t farea, iarea, offset;
    size_t row, col, depth;
    size_t image_offset;
    size_t sample_offset;

    farea = (size_t)filter_depth * (size_t)filter_rows * (size_t)filter_cols *
            (size_t)output_depth * (size_t)output_rows * (size_t)output_cols * (size_t)count;

    iarea = input_depth * input_rows * input_cols;

#pragma omp parallel for private(j,k,l,m,n,o,p,depth,row,col,image_offset,sample_offset,offset) schedule(dynamic)
    for(i=0; i<input_channels; i++){
		offset = i*farea;
        for(j=0; j<filter_depth; j++){
            for(k=0; k<filter_rows; k++){
                for(l=0; l<filter_cols; l++){
                    for(m=0; m<count; m++){
					    sample_offset = (m * input_channels + i) * iarea;
                        depth = j - pad_depth;
                        for(n=0; n<output_depth; n++){
                            if(is_a_ge_zero_and_a_lt_b(depth, input_depth)){
                                row = k - pad_rows;
                                for(o=0; o<output_rows; o++){
                                    if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
                                        col = l - pad_cols;
                                        for(p=0; p<output_cols; p++){
                                            if(is_a_ge_zero_and_a_lt_b(col, input_cols)){
                                                image_offset = depth * input_rows * input_cols + row * input_cols + col;
									            column_buffer[offset++] = data[sample_offset + image_offset];
                                            }
                                            else{
									            column_buffer[offset++] = 0;
                                            }
                                            col += stride_cols;
                                        }
                                    }
                                    else{
                                        for(p=0; p<output_cols; p++){
                                            column_buffer[offset++] = 0;
                                        }
                                    }
                                    row += stride_rows;
                                }
                            }
                            else{
                                for(o=0; o<output_rows; o++){
                                    for(p=0; p<output_cols; p++){
								        column_buffer[offset++] = 0;
                                    }
                                }
                            }
                            depth += stride_depth;
                        }
                    }
                }
            }
        }
    }
}

/* Input: (IF) x (N x ID x IR x IC)
 * Output: (IF x FD x FR x FC) x (N x OD x OR x OC) */
void pcnn_transform_im2col2(float *data, float *column_buffer, int count,
                            int pad_depth, int pad_rows, int pad_cols,
                            int stride_depth, int stride_rows, int stride_cols,
                            int input_channels, int input_depth, int input_rows, int input_cols,
                            int output_depth, int output_rows, int output_cols,
                            int filter_depth, int filter_rows, int filter_cols)
{
    size_t i,j,k,l,m,n,o,p;
    size_t farea, iarea, offset;
    size_t depth, row, col;
    size_t image_offset;
    size_t sample_offset;

    farea = (size_t)filter_depth * (size_t)filter_rows * (size_t)filter_cols *
            (size_t)output_depth * (size_t)output_rows * (size_t)output_cols * (size_t)count;

    iarea = input_depth * input_rows * input_cols;

#pragma omp parallel for private(j,k,l,m,n,o,p,depth,row,col,image_offset,sample_offset,offset) schedule(dynamic)
    for(i=0; i<input_channels; i++){
		offset = i*farea;
        for(j=0; j<filter_depth; j++){
            for(k=0; k<filter_rows; k++){
                for(l=0; l<filter_cols; l++){
                    for(m=0; m<count; m++){
					    sample_offset = (i * count + m) * iarea;
                        depth = j - pad_depth;
                        for(n=0; n<output_depth; n++){
                            if(is_a_ge_zero_and_a_lt_b(depth, input_depth)){
                                row = k - pad_rows;
                                for(o=0; o<output_rows; o++){
                                    if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
                                        col = l - pad_cols;
                                        for(p=0; p<output_cols; p++){
                                            if(is_a_ge_zero_and_a_lt_b(col, input_cols)){
                                                image_offset = depth * input_rows * input_cols + row * input_cols + col;
									            column_buffer[offset++] = data[sample_offset + image_offset];
                                            }
                                            else{
									            column_buffer[offset++] = 0;
                                            }
                                            col += stride_cols;
                                        }
                                    }
                                    else{
                                        for(p=0; p<output_cols; p++){
                                            column_buffer[offset++] = 0;
                                        }
                                    }
                                    row += stride_rows;
                                }
                            }
                            else{
                                for(o=0; o<output_rows; o++){
                                    for(p=0; p<output_cols; p++){
								        column_buffer[offset++] = 0;
                                    }
                                }
                            }
                            depth += stride_depth;
                        }
                    }
                }
            }
        }
    }
}

/* Input: (IF) x (N x ID x IR x IC)
 * Output: (N x OD x OR x OC) x (IF x FD x FR x FC) */
void pcnn_transform_im2col_prime(float *data, float *column_buffer, int count,
                                 int pad_depth, int pad_rows, int pad_cols,
                                 int stride_depth, int stride_rows, int stride_cols,
                                 int input_channels, int input_depth, int input_rows, int input_cols,
                                 int output_depth, int output_rows, int output_cols,
                                 int filter_depth, int filter_rows, int filter_cols)
{
    size_t i,j,k,l,m,n,o,p;
    size_t iarea, rarea, offset;
    size_t depth, row, col;
    size_t image_offset;
    size_t sample_offset;
    size_t off_row, off_col;
    size_t off_rows, off_cols;

    iarea = input_depth * input_rows * input_cols;
    rarea = input_channels * filter_depth * filter_rows * filter_cols;

#pragma omp parallel for private(j,k,l,m,n,o,p,depth,row,col,image_offset,sample_offset,offset,off_rows,off_cols) schedule(dynamic)
    for(i=0; i<input_channels; i++){
		off_cols = i * filter_depth * filter_rows * filter_cols;
        for(j=0; j<filter_depth; j++){
            for(k=0; k<filter_rows; k++){
                for(l=0; l<filter_cols; l++){
                    off_rows =0;
                    for(m=0; m<count; m++){
					    sample_offset = (i * count + m) * iarea;
                        depth = j - pad_depth;
                        for(n=0; n<output_depth; n++){
                            if(is_a_ge_zero_and_a_lt_b(depth, input_depth)){
                                row = k - pad_rows;
                                for(o=0; o<output_rows; o++){
                                    if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
                                        col = l - pad_cols;
                                        for(p=0; p<output_cols; p++){
                                            if(is_a_ge_zero_and_a_lt_b(col, input_cols)){
                                                image_offset = depth * input_rows * input_cols + row * input_cols + col;
                                                offset = off_rows * rarea + off_cols;
									            column_buffer[offset] = data[sample_offset + image_offset];
                                                off_rows++;
                                            }
                                            else{
                                                offset = off_rows * rarea + off_cols;
									            column_buffer[offset] = 0;
                                                off_rows++;
                                            }
                                            col += stride_cols;
                                        }
                                    }
                                    else{
                                        for(p=0; p<output_cols; p++){
                                            offset = off_rows * rarea + off_cols;
                                            column_buffer[offset] = 0;
                                            off_rows++;
                                        }
                                    }
                                    row += stride_rows;
                                }
                            }
                            else{
                                for(o=0; o<output_rows; o++){
                                    for(p=0; p<output_cols; p++){
                                        offset = off_rows * rarea + off_cols;
								        column_buffer[offset] = 0;
                                        off_rows++;
                                    }
                                }
                            }
                            depth += stride_depth;
                        }
                    }
                    off_cols++;
                }
            }
        }
    }
}

/* Input: (IF x FD x FR x FC) x (N x OD x OR x OC)
 * Output: (IF) x (N x ID x IR x IC) */
void pcnn_transform_col2im(float *data, float *column_buffer, int count,
                           int pad_depth, int pad_rows, int pad_cols,
                           int stride_depth, int stride_rows, int stride_cols,
                           int input_channels, int input_depth, int input_rows, int input_cols,
                           int output_depth, int output_rows, int output_cols,
                           int filter_depth, int filter_rows, int filter_cols)
{
    size_t i,j,k,l,m,n,o,p;
    size_t depth, row, col;
    size_t in_off, out_off, offset;
    const size_t iarea = input_depth * input_rows * input_cols;
    const size_t area = (size_t)count * (size_t)output_depth * (size_t)output_rows * (size_t)output_cols *
                        (size_t)filter_depth * (size_t)filter_rows * (size_t)filter_cols;

#pragma omp parallel for private(j,k,l,m,n,o,p,depth,row,col,in_off,out_off,offset)
	for(i=0; i<input_channels; i++){
		in_off = i*area;
        for(j=0; j<filter_depth; j++){
            for(k=0; k<filter_rows; k++){
                for(l=0; l<filter_cols; l++){
                    for(m=0; m<count; m++){
                        out_off = (i * count + m) * iarea;
                        depth = j - pad_depth;
                        for(n=0; n<output_depth; n++){
                            if(is_a_ge_zero_and_a_lt_b(depth, input_depth)){
                                row = k - pad_rows;
                                for(o=0; o<output_rows; o++){
                                    if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
                                        col = l - pad_cols;
                                        for(p=0; p<output_cols; p++){
                                            if(is_a_ge_zero_and_a_lt_b(col, input_cols)){
                                                offset = out_off + (depth * input_rows * input_cols) + (row * input_cols) + col;
                                                data[offset] += column_buffer[in_off];
                                            }
                                            in_off++;
                                            col += stride_cols;
                                        }
                                    }
                                    else{
                                        in_off += output_cols;
                                    }
                                    row += stride_rows;
                                }
                            }
                            else{
                                in_off += (output_rows * output_cols);
                            }
                            depth += stride_depth;
                        }
                    }
                }
            }
        }
	}
}

/* Transform the data after MPI_Alltoall communication.
 * source: PN x (K/P) 
 * destination: (PN/P) x K
 * PN -> rows
 * K -> cols
 * P -> numGroups
 * Assuming rows and cols are divisible by numGroups.
 */
int pcnn_transform_rearrange(float *src, float *dest, int rows, int cols, int num_procs)
{
	size_t i,j,k;
	size_t area, local_rows, local_cols;

	local_rows = rows / num_procs;
	local_cols = cols / num_procs;
	area = local_rows*local_cols;
	
#pragma omp parallel for private(j,k)
	for(i=0; i<num_procs; i++){
		for(j=0; j<local_rows; j++){
			for(k=0; k<local_cols; k++){
				dest[j*cols + i*local_cols + k] = src[i*area + j*local_cols + k];
			}
		}
	}
	
	return 0;
}
