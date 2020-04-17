/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <boost/random.hpp>
#include "def.h"
#include "model.h"
#include "feeder.h"
#include "util.h"
#include "comm.h"

float sigmoid(float t)
{
	return (float)1/(float)(1+exp(-t));
}

float sigmoid_prime(float t)
{
	float e = sigmoid(t);
	return (float)e*(float)((float)1-(float)e);
}

float randn(float mu, float sigma)
{
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (float) X2);
    }
 
  do
    {
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (float) X1);
}

static void swap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

int pcnn_util_shuffle(int *array, int num)
{
	int i=0;
	int j=0;
	
	srand(time(NULL));
		
	for(i=num-1; i>0; i--){
		j = rand() % (i+1);
		swap(&array[i], &array[j]);
	}
	return 0;
}

void pcnn_util_gaussian(int seed, const int length, const float mean, const float sigma, float *data)
{
	int i;

	srand(time(NULL) + seed);
	boost::normal_distribution<float> nd(mean, sigma);
	boost::mt19937 rgn(rand());
	boost::variate_generator<boost::mt19937, boost::normal_distribution<float> > generator(rgn, nd);

	for(i=0; i<length; i++)
		data[i] = generator();
}

float pcnn_util_calc_PSNR(float *image1, float *image2, int num_pixels)
{
    int i = 0;
    float sum = 0;
    float PSNR = 0;

#pragma omp parallel for reduction(+:sum)
    for(i=0; i<num_pixels; i++){
        sum += (image1[i] - image2[i]) * (image1[i] - image2[i]);
    }

    sum /= num_pixels;
    PSNR = 20 * log10f(255.f);
    PSNR -= 10 * log10f(sum);
    return PSNR;
}

void pcnn_util_evaluate(int imgidx, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int i,j,k,l;
    int area;
    int src_off, dst_off;
    int max_neuron;
    float *image = NULL;
    float sum, max_value;
    struct layer_t *layer;

    layer = model->layers[model->num_layers-1];
    if(layer->type == LAYER_TYPE_CONV || layer->type == LAYER_TYPE_UPSAMPLE){
        area = layer->output_rows * layer->output_cols;
        image = (float *)malloc(sizeof(float) * layer->output_channels * area);
        sum = 0;
        for(i=0; i<feeder->local_batch_size; i++){
            dst_off = 0;
            for(j=0; j<layer->output_channels; j++){
                src_off = (j * feeder->local_batch_size + i) * area;
                for(k=0; k<layer->output_rows; k++){
                    for(l=0; l<layer->output_cols; l++){
                        image[dst_off] = layer->a[src_off++];
                        if (image[dst_off] < 0) image[dst_off] = 0.0f;
                        if (image[dst_off] > 255) image[dst_off] = 255.0f;
                        dst_off++;
                    }
                }
            }
            sum += pcnn_util_calc_PSNR(image, &feeder->label[(imgidx + i) * feeder->label_size], feeder->label_size);
        }
        param->custom_output += (sum / feeder->batch_size);
        free(image);
    }
    else if(layer->type == LAYER_TYPE_FULL){
        if(model->task_type == TASK_TYPE_CLASSIFICATION){
            for(i=0; i<feeder->local_batch_size; i++){
                max_value = layer->a[i];
                max_neuron = 0;
                for(j=1; j<feeder->label_size; j++){
                    if(max_value < layer->a[j * feeder->local_batch_size + i]){
                        max_value = layer->a[j * feeder->local_batch_size + i];
                        max_neuron = j;
                    }
                }
                if(feeder->label[(imgidx + i) * feeder->label_size + max_neuron])
                    param->num_corrects += 1;
            }
        }
        else if(model->task_type == TASK_TYPE_REGRESSION){
            /* We use loss as the evaluation metric for regression problems. */
        }
    }
}
