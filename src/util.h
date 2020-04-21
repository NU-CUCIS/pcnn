/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
int pcnn_util_shuffle(int *array, int num);
int pcnn_util_transform(float *src, float *dest, int rows, int cols, int numGroups);
void pcnn_util_gaussian(int seed, const int length, const float mean, const float sigma, float *data);
float pcnn_util_calc_PSNR(float *image1, float *image2, int num_pixels);
void pcnn_util_evaluate(struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
