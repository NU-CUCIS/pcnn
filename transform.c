/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>

/* This function is brought from Caffe source code (caffe/util/im2col.cpp) */
inline int is_a_ge_zero_and_a_lt_b(int a, int b){
	return (unsigned)(a) < (unsigned)(b);
}

/* Input: (N) x (ID x IR x IC)
 * Output: (ID x FR x FC) x (N x OR x OC) */
void pcnn_transform_im2col1(float *data, float *column_buffer, 
                            int pad_rows, int pad_cols,
                            int stride_rows, int stride_cols,
                            int input_depth, int input_rows, int input_cols,
                            int output_rows, int output_cols,
                            int filter_rows, int filter_cols, int count)
{
	int i,j,k,l,m,n;
	int farea, offset;
	int row, col;
	int image_offset=0;

	farea = filter_rows * filter_cols * count * output_rows * output_cols;

#pragma omp parallel for private(j,k,l,m,n,offset,row,col,image_offset) schedule(dynamic)
	for(i=0; i<input_depth; i++){
		offset = i*farea;
		for(j=0; j<filter_rows; j++){
			for(k=0; k<filter_cols; k++){
				for(l=0; l<count; l++){
					image_offset = (l*input_depth+i)*input_rows*input_cols;
					row = j - pad_rows;
					for(m=0; m<output_rows; m++){
						if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
							col = k - pad_cols;
							for(n=0; n<output_cols; n++){
								if(is_a_ge_zero_and_a_lt_b(col, input_cols))
									column_buffer[offset++] = data[image_offset+row*input_cols+col];
								else
									column_buffer[offset++] = 0;

								col += stride_cols;
							}
						}
						else{
							for(n=0; n<output_cols; n++)
								column_buffer[offset++] = 0;
						}
						row += stride_rows;
					}
				}
			}
		}
	}
}

/* Input: (ID) x (N x IR x IC)
 * Output: (ID x FR x FC) x (N x OR x OC) */
void pcnn_transform_im2col2(float *data, float *column_buffer, 
                            int pad_rows, int pad_cols,
                            int stride_rows, int stride_cols,
                            int input_depth, int input_rows, int input_cols,
                            int output_rows, int output_cols,
                            int filter_rows, int filter_cols, int count)
{
	int i,j,k,l,m,n;
	unsigned int farea, offset;
	unsigned int row, col;
	unsigned int image_offset=0;

	farea = filter_rows * filter_cols * count * output_rows * output_cols;

#pragma omp parallel for private(j,k,l,m,n,offset,row,col,image_offset) schedule(dynamic)
	for(i=0; i<input_depth; i++){
		offset = i*farea;
		for(j=0; j<filter_rows; j++){
			for(k=0; k<filter_cols; k++){
				for(l=0; l<count; l++){
					image_offset = (i*count+l)*input_rows*input_cols;
					row = j - pad_rows;
					for(m=0; m<output_rows; m++){
						if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
							col = k - pad_cols;
							for(n=0; n<output_cols; n++){
								if(is_a_ge_zero_and_a_lt_b(col, input_cols))
									column_buffer[offset++] = data[image_offset+row*input_cols+col];
								else
									column_buffer[offset++] = 0;

								col += stride_cols;
							}
						}
						else{
							for(n=0; n<output_cols; n++)
								column_buffer[offset++] = 0;
						}
						row += stride_rows;
					}
				}
			}
		}
	}
}

/* Input: (ID) x (N x IR x IC)
 * Output: (N x OR x OC) x (ID x FR x FC) */
void pcnn_transform_im2col_prime(float *data, float *column_buffer, 
                                 int pad_rows, int pad_cols,
                                 int stride_rows, int stride_cols,
                                 int input_depth, int input_rows, int input_cols,
                                 int output_rows, int output_cols,
                                 int filter_rows, int filter_cols, int count)
{
	int i,j,k,l,m,n;
	unsigned int farea, offset;
	unsigned int row, col;
	unsigned int image_offset=0;
    unsigned int off_row, off_col;

    farea = input_depth * filter_rows * filter_cols;
#pragma omp parallel for private(j,k,l,m,n,offset,off_row,off_col,row,col,image_offset) schedule(dynamic)
	for(i=0; i<input_depth; i++){
        off_col = i*filter_rows*filter_cols;
		for(j=0; j<filter_rows; j++){
			for(k=0; k<filter_cols; k++){
                off_row = 0;
				for(l=0; l<count; l++){
					image_offset = (i*count+l)*input_rows*input_cols;
					row = j - pad_rows;
					for(m=0; m<output_rows; m++){
						if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
							col = k - pad_cols;
							for(n=0; n<output_cols; n++){
                                offset = off_row*farea + off_col;
								if(is_a_ge_zero_and_a_lt_b(col, input_cols)){
									column_buffer[offset] = data[image_offset+row*input_cols+col];
                                }
								else{
									column_buffer[offset] = 0;
                                }
								col += stride_cols;
                                off_row++;
							}
						}
						else{
							for(n=0; n<output_cols; n++){
                                offset = off_row*farea + off_col;
								column_buffer[offset] = 0;
                                off_row++;
                            }
						}
						row += stride_rows;
					}
				}
                off_col++;
			}
		}
	}
}

/* Input: (ID x FR x FC) x (N x OR x OC)
 * Output: (ID) x (N x IR x IC) */
void pcnn_transform_col2im(float *data, float *column_buffer,
                           int pad_rows, int pad_cols,
                           int stride_rows, int stride_cols,
                           int input_depth, int input_rows, int input_cols,
                           int output_rows, int output_cols,
                           int filter_rows, int filter_cols, int count)
{
	int i,j,k,l,m,n;
	unsigned int row, col;
	unsigned int in_off, out_off;
	const unsigned int area = count * output_rows * output_cols * filter_rows * filter_cols;

#pragma omp parallel for private(j,k,l,m,n,row,col,in_off,out_off)
	for(i=0; i<input_depth; i++){
		in_off = i*area;
		for(j=0; j<filter_rows; j++){
			for(k=0; k<filter_cols; k++){
				for(l=0; l<count; l++){
					row = j - pad_rows;
					out_off = (i*count+l)*input_rows*input_cols;
					for(m=0; m<output_rows; m++){
						if(is_a_ge_zero_and_a_lt_b(row, input_rows)){
							col = k - pad_cols;
							for(n=0; n<output_cols; n++){
								if(is_a_ge_zero_and_a_lt_b(col, input_cols))
									data[out_off + row*input_cols+col] += column_buffer[in_off];
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
	int i,j,k;
	int area, local_rows, local_cols;

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
