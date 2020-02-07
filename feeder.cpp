/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#ifdef USE_MKL 
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include "def.h"
#include "config.h"
#include "model.h"
#include "feeder.h"
#include "util.h"
#include "comm.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

/* Feeder module should be modified by users based on the training dataset.
 * A few examples have been implemented such as mnist, CIFAR10, div2k, and ImageNet.
 * Note that this code is basically in C, but to use OpenCV which is C++ library,
 * some functions use C++ features and this file becomes .cpp.
 */

/* Static functions */
static int pcnn_feeder_mnist_init(struct feeder_t *feeder);
static int pcnn_feeder_cifar10_init(struct feeder_t *feeder);
static int pcnn_feeder_imagenet_init(struct feeder_t *feeder);
static int pcnn_feeder_phantom_init(struct feeder_t *feeder);
static int pcnn_feeder_div2k_init(struct feeder_t *feeder);
static int pcnn_feeder_ghost_init(struct feeder_t *feeder);
static int pcnn_feeder_read_meanfile(struct feeder_t *feeder);
static int pcnn_feeder_read_jpeg(int test, int image_index, char **image_paths, char **label_paths, struct feeder_t *feeder);
static int pcnn_feeder_get_batch_memory(int test, int batch_index, struct feeder_t *feeder, struct comm_queue_t *queue);
static int pcnn_feeder_get_batch_imagenet(int test, int batch_index, struct feeder_t *feeder, struct comm_queue_t *queue);
static int pcnn_feeder_get_batch_phantom(int test, int batch_index, struct feeder_t *feeder);
static int pcnn_feeder_get_batch_div2k(int test, int batch_index, int upsample_ratio, struct feeder_t *feeder, struct comm_queue_t *queue);
static int pcnn_feeder_get_ghost_batch(int test, int batch_index, struct feeder_t *feeder);

struct feeder_t *pcnn_feeder_init(int do_shuffle, struct comm_queue_t *queue)
{
    int i, ret = 0;
    struct feeder_t *feeder = NULL;

    feeder = (struct feeder_t *)calloc(1, sizeof(struct feeder_t));
    feeder->do_shuffle = do_shuffle;

    /* Settings from def.h */
    feeder->do_crop = CROP_IMAGES;
    feeder->batch_size = BATCH_SIZE;
#if MNIST
    feeder->dataset = DATASET_MNIST;
#elif CIFAR10
    feeder->dataset = DATASET_CIFAR10;
#elif IMAGENET
    feeder->dataset = DATASET_IMAGENET;
#elif PHANTOM
    feeder->dataset = DATASET_PHANTOM;
#elif DIV2K
    feeder->dataset = DATASET_DIV2K;
#elif GHOST_BATCH
    feeder->dataset = DATASET_GHOST;
#endif

    /* Check if batch size is valid. */
    if(feeder->batch_size % queue->nproc != 0){
        printf("[%s][%d] Batch size should be divisible by the number of processes!\n", __FUNCTION__, __LINE__);
        free(feeder);
        return NULL;
    }

    if(feeder->do_crop == 1 && feeder->dataset < DATASET_PHANTOM){
        printf("Currently, image cropping is supported for phantom dataset only!\n");
        free(feeder);
        return NULL;
    }

    feeder->local_batch_size = feeder->batch_size / queue->nproc;
    if(feeder->batch_size % queue->nproc > 0){
        printf("[%s][%d] The mini-batch size should be divisible by the number of processes!\n", __FUNCTION__, __LINE__);
        free(feeder);
        return NULL;
    }

    feeder->crop_offset_x = NULL;
    feeder->crop_offset_y = NULL;
    feeder->train_offset = NULL;
    feeder->large_frame = NULL;

    if(feeder->dataset == DATASET_MNIST){
        /* binary images and labels */
        ret = pcnn_feeder_mnist_init(feeder);
    }
    else if(feeder->dataset == DATASET_CIFAR10){
        /* binary images and labels */
        ret = pcnn_feeder_cifar10_init(feeder);
    }
    else if(feeder->dataset == DATASET_IMAGENET){
        /* JPEG images and a label file */
        ret = pcnn_feeder_imagenet_init(feeder);
    }
    else if(feeder->dataset == DATASET_PHANTOM){
        /* PNG/JPEG images and PNG/JPEG labels */
        ret = pcnn_feeder_phantom_init(feeder);
    }
    else if(feeder->dataset == DATASET_DIV2K){
        /* PNG/JPEG images and PNG/JPEG labels */
        ret = pcnn_feeder_div2k_init(feeder);
    }
    else if(feeder->dataset == DATASET_GHOST){
        /* Empty batch for measuring the performance. */
        ret = pcnn_feeder_ghost_init(feeder);
    }
    if(ret != 0){
        printf("[%s][%d] pcnn_feeder_XXX_init failed!\n", __FUNCTION__, __LINE__);
        free(feeder);
        return NULL;
    }

    feeder->train_order = (int *)malloc(sizeof(int)*feeder->num_train_images);
    for(i=0; i<feeder->num_train_images; i++)
        feeder->train_order[i] = i;

    if(feeder->dataset == DATASET_MNIST){
        /* MNIST is a binary dataset. We do not shuffle every single image.
         * When reading each batch, every process reads an entire batch from 
         * the offset. So, the index value should not exceed N - B. */
        for(i=feeder->num_train_images - feeder->batch_size; i<feeder->num_train_images; i++)
            feeder->train_order[i] = feeder->num_train_images - feeder->batch_size;
    }

    ret = pcnn_feeder_read_meanfile(feeder);
    if(ret != 0){
        printf("[%s][%d] pcnn_feeder_read_meanfile failed!\n", __FUNCTION__, __LINE__);
        free(feeder);
        return NULL;
    }
    
    if(queue->group_id == 0 && queue->rank == 0){
        printf("---------------------------------------------------------\n");
        printf("%-40s: %d\n", "Number of training images", feeder->num_train_images);
        printf("%-40s: %d\n", "Size of mini-batch", feeder->batch_size);
        printf("%-40s: %d\n", "Size of local mini-batch", feeder->local_batch_size);
        printf("%-40s: %d\n", "Shuffle the training data", feeder->do_shuffle);
        printf("%-40s: %d x %d x %d\n", "Original image size", feeder->image_depth, feeder->image_orig_width, feeder->image_orig_height);
        printf("%-40s: %d x %d x %d\n", "Cropped image size", feeder->image_depth, feeder->image_width, feeder->image_height);
    }
    return feeder;
}

void pcnn_feeder_destroy(struct feeder_t *feeder)
{
	if(feeder != NULL){
        if(feeder->large_frame != NULL)
            free(feeder->large_frame);
        if(feeder->mean_image != NULL)
            free(feeder->mean_image);
		if(feeder->minibatch != NULL)
			free(feeder->minibatch);
		if(feeder->label != NULL)
			free(feeder->label);
        if(feeder->crop_offset_x != NULL)
            free(feeder->crop_offset_x);
        if(feeder->crop_offset_y != NULL)
            free(feeder->crop_offset_y);
		if(feeder->train_offset != NULL)
			free(feeder->train_offset);
		if(feeder->test_offset != NULL)
			free(feeder->test_offset);
		if(feeder->train_order != NULL)
			free(feeder->train_order);
		if(feeder->test_order != NULL)
			free(feeder->test_order);
		free(feeder);
	}
}

int pcnn_feeder_get_minibatch(int test, int batch_index, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue)
{
	int ret = 0;

    if(feeder->dataset == DATASET_MNIST){
        ret = pcnn_feeder_get_batch_memory(test, batch_index, feeder, queue);
    }
    else if(feeder->dataset == DATASET_CIFAR10){
        ret = pcnn_feeder_get_batch_memory(test, batch_index, feeder, queue);
    }
    else if(feeder->dataset == DATASET_IMAGENET){
        ret = pcnn_feeder_get_batch_imagenet(test, batch_index, feeder, queue);
    }
    else if(feeder->dataset == DATASET_PHANTOM){
        ret = pcnn_feeder_get_batch_phantom(test, batch_index, feeder);
    }
    else if(feeder->dataset == DATASET_DIV2K){
        ret = pcnn_feeder_get_batch_div2k(test, batch_index, model->upsample_ratio, feeder, queue);
    }
    else if(feeder->dataset == DATASET_GHOST){
        ret = pcnn_feeder_get_ghost_batch(test, batch_index, feeder);
    }
    else{
        ret = -1;
    }

	return ret;
}

void pcnn_feeder_shuffle(struct feeder_t *feeder, struct comm_queue_t *queue)
{
	pcnn_util_shuffle(feeder->train_order, feeder->num_train_images);
	MPI_Bcast(feeder->train_order, feeder->num_train_images, MPI_INT, 0, queue->world);
}

void pcnn_feeder_subtract_mean_image(struct feeder_t *feeder)
{
    int i,j;
    float pixel_value;

    if(feeder->dataset == DATASET_MNIST || 
       feeder->dataset == DATASET_PHANTOM)
        return;

#pragma omp parallel for private(j, pixel_value)
    for(i=0; i<feeder->local_batch_size; i++){
        for(j=0; j<feeder->num_pixels; j++){
            pixel_value = static_cast<float>(static_cast<unsigned char>(feeder->minibatch[i*feeder->num_pixels+j]));
            feeder->minibatch[i*feeder->num_pixels+j] = pixel_value - feeder->mean_image[j];
        }
    }
}

/*******************************************
 * Static functions
 ******************************************/
static int pcnn_feeder_mnist_init(struct feeder_t *feeder)
{
    int i,j;
    int label;
    int type;
    int num_rows;
    int num_cols;
    int num_images;
    size_t ret;
    FILE *fd=NULL;
    char *temp=NULL;
    char path[200];

    sprintf(feeder->train_top_dir, MNIST_TRAIN_TOP_DIR);
    sprintf(feeder->test_top_dir, MNIST_TEST_TOP_DIR);

    /* Read the entire training and test datasets into memory. */
    /* Training set image file */
    sprintf(path, "%s/%s", MNIST_TRAIN_TOP_DIR, MNIST_TRAIN_IMAGE);
    fd = fopen(path, "r");
    if(fd == NULL){
        printf("[%s][%d] opening %s failed.\n", __FUNCTION__, __LINE__, path);
        return -1;
    }

    ret = fread(&type, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_images, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_rows, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_cols, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    /* settings from def.h */
    feeder->image_depth = MNIST_DEPTH;
    feeder->image_width = MNIST_WIDTH;
    feeder->image_height = MNIST_HEIGHT;
    feeder->image_orig_width = MNIST_WIDTH;
    feeder->image_orig_height = MNIST_HEIGHT;
    feeder->num_train_images = __bswap_32(num_images);
    feeder->num_pixels = __bswap_32(num_rows) * __bswap_32(num_cols);
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;

	feeder->binary_train_data = (float *)malloc(sizeof(float)*feeder->num_train_images*feeder->num_pixels);
    temp = (char *)malloc(sizeof(char)*feeder->num_train_images*feeder->num_pixels);

    ret = fread(temp, sizeof(char), feeder->num_train_images*feeder->num_pixels, fd);
    if(ret != (size_t)(feeder->num_train_images*feeder->num_pixels)){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    for(i=0; i<feeder->num_train_images; i++){
        for(j=0; j<feeder->num_pixels; j++){
            feeder->binary_train_data[i*feeder->num_pixels+j] = static_cast<float>(temp[i*feeder->num_pixels+j]);
        }
    }
    free(temp);
    fclose(fd);

    /* Training set label file */
    sprintf(path, "%s/%s", MNIST_TRAIN_TOP_DIR, MNIST_TRAIN_LABEL);
    fd = fopen(path, "r");
    if(fd == NULL){
        printf("[%s][%d] opening %s failed.\n", __FUNCTION__, __LINE__, path);
        return -1;
    }
    ret = fread(&type, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_images, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    feeder->label_size = MNIST_LABEL_SIZE;

	feeder->binary_train_label = (float *)calloc(feeder->num_train_images*feeder->label_size, sizeof(float));
    temp = (char *)malloc(sizeof(char)*feeder->num_train_images);

    ret = fread(temp, sizeof(char), feeder->num_train_images, fd);
    if(ret != size_t(feeder->num_train_images)){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }
    
    for(i=0; i<feeder->num_train_images; i++){
        label = static_cast<int>(temp[i]);
        feeder->binary_train_label[i*feeder->label_size+label] = 1; 
    }
    free(temp);
    fclose(fd);

    /* Teet set image file */
    sprintf(path, "%s/%s", MNIST_TEST_TOP_DIR, MNIST_TEST_IMAGE);
    fd = fopen(path, "r");
    if(fd == NULL){
        printf("[%s][%d] opening %s failed.\n", __FUNCTION__, __LINE__, path);
        return -1;
    }
    ret = fread(&type, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_images, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_rows, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_cols, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    feeder->num_test_images   = __bswap_32(num_images);
	feeder->num_test_batches  = feeder->num_test_images / feeder->batch_size;

	feeder->binary_test_data = (float *)malloc(sizeof(float)*feeder->num_test_images*feeder->num_pixels);
    temp = (char *)malloc(sizeof(char)*feeder->num_test_images*feeder->num_pixels);

    ret = fread(temp, sizeof(char), feeder->num_test_images*feeder->num_pixels, fd);
    if(ret != size_t(feeder->num_test_images*feeder->num_pixels)){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    for(i=0; i<feeder->num_test_images; i++){
        for(j=0; j<feeder->num_pixels; j++){
            feeder->binary_test_data[i*feeder->num_pixels+j] = static_cast<float>(temp[i*feeder->num_pixels+j]);
        }
    }
    free(temp);
    fclose(fd);

    /* Teet set label file */
    sprintf(path, "%s/%s", MNIST_TEST_TOP_DIR, MNIST_TEST_LABEL);
    fd = fopen(path, "r");
    if(fd == NULL){
        printf("[%s][%d] opening %s failed.\n", __FUNCTION__, __LINE__, path);
        return -1;
    }
    ret = fread(&type, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

    ret = fread(&num_images, sizeof(int), 1, fd);
    if(ret != 1){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }

	feeder->binary_test_label = (float *)calloc(feeder->num_test_images*feeder->label_size, sizeof(float));
    temp = (char *)malloc(sizeof(char)*feeder->num_test_images);

    ret = fread(temp, sizeof(char), feeder->num_test_images, fd);
    if(ret != size_t(feeder->num_test_images)){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        return -1;
    }
    
    for(i=0; i<feeder->num_test_images; i++){
        label = static_cast<int>(temp[i]);
        feeder->binary_test_label[i*feeder->label_size+label] = 1; 
    }
    free(temp);
    fclose(fd);

    /* Memory allocation for minibatch and labels. */
	feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
	feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);

    return 0;
}

static int pcnn_feeder_cifar10_init(struct feeder_t *feeder)
{
    int i, j, k, offset;
    int sub_file_size=10000;
    int num_sub_files=5;
    int label_offset;
    int data_offset;
    int label_size, data_size, total_size;
    unsigned char *temp;
    char **files;
    unsigned char label;
    FILE *fd;
    size_t ret;

    feeder->num_train_images = 50000;
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;

    /* settings from def.h */
    feeder->image_depth = CIFAR10_DEPTH;
    feeder->image_width = CIFAR10_WIDTH;
    feeder->image_height = CIFAR10_HEIGHT;
    feeder->image_orig_width = CIFAR10_WIDTH;
    feeder->image_orig_height = CIFAR10_HEIGHT;
    feeder->label_size = CIFAR10_LABEL_SIZE;
    feeder->large_frame = (float *)calloc(CIFAR10_DEPTH * (CIFAR10_HEIGHT + 8) * (CIFAR10_WIDTH + 8), sizeof(float));

    feeder->num_pixels = feeder->image_depth * feeder->image_width * feeder->image_height;
    sprintf(feeder->train_top_dir, CIFAR10_TRAIN_TOP_DIR);
    sprintf(feeder->test_top_dir, CIFAR10_TEST_TOP_DIR);

    label_size = sub_file_size;
    data_size = sub_file_size * feeder->num_pixels;
    total_size = label_size + data_size;

    /* Memory allocation for training data. */
    temp = (unsigned char *)malloc(total_size * sizeof(unsigned char));
    feeder->binary_train_data = (float *)malloc(sizeof(float)*feeder->num_train_images*feeder->num_pixels);
    feeder->binary_train_label = (float *)malloc(sizeof(float)*feeder->num_train_images*feeder->label_size);
    memset(feeder->binary_train_label, 0, sizeof(float)*feeder->num_train_images*feeder->label_size);

    files = (char **)malloc(sizeof(char *)*num_sub_files);
    files[0] = (char *)malloc(sizeof(char)*num_sub_files*100);
    for(i=1; i<num_sub_files; i++)
        files[i] = files[i-1] + 100;

    sprintf(files[0], "%s/%s", CIFAR10_TRAIN_TOP_DIR, CIFAR10_TRAIN_IMAGE1);
    sprintf(files[1], "%s/%s", CIFAR10_TRAIN_TOP_DIR, CIFAR10_TRAIN_IMAGE2);
    sprintf(files[2], "%s/%s", CIFAR10_TRAIN_TOP_DIR, CIFAR10_TRAIN_IMAGE3);
    sprintf(files[3], "%s/%s", CIFAR10_TRAIN_TOP_DIR, CIFAR10_TRAIN_IMAGE4);
    sprintf(files[4], "%s/%s", CIFAR10_TRAIN_TOP_DIR, CIFAR10_TRAIN_IMAGE5);

    /* Read the entire binary training data. */
    for(i=0; i<num_sub_files; i++){
        fd = fopen(files[i], "r");
        ret = fread(temp, sizeof(unsigned char), total_size, fd);
        if(ret != (size_t)total_size * sizeof(unsigned char)){
            printf("[%s][%d] fread failed. ret: %lu\n", __FUNCTION__, __LINE__, ret);
            fclose(fd);
            return -1;
        }
        fclose(fd);

        offset = 0;
        for(j=0; j<sub_file_size; j++){
            label = temp[offset++];
            label_offset = (i*sub_file_size+j)*feeder->label_size + (int)label;
            feeder->binary_train_label[label_offset] = 1;

            for(k=0; k<feeder->num_pixels; k++){
                data_offset = (i*sub_file_size+j)*feeder->num_pixels + k;
                feeder->binary_train_data[data_offset] = static_cast<float>(static_cast<unsigned char>(temp[offset++]));
            }
        }
    }

    /* Memory allocation for test data. */
    feeder->num_test_images = 10000;
    feeder->num_test_batches = feeder->num_test_images / feeder->batch_size;
    feeder->binary_test_data = (float *)malloc(sizeof(float)*feeder->num_test_images*feeder->num_pixels);
    feeder->binary_test_label = (float *)malloc(sizeof(float)*feeder->num_test_images*feeder->label_size);
    memset(feeder->binary_test_label, 0, sizeof(float)*feeder->num_test_images*feeder->label_size);

    sprintf(files[0], "%s/%s", CIFAR10_TEST_TOP_DIR, CIFAR10_TEST_IMAGE);

    /* Read the binary test data. */
    fd = fopen(files[0], "r");
    ret = fread(temp, sizeof(unsigned char), total_size, fd);
    if(ret != (size_t)total_size * sizeof(unsigned char)){
        printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
        fclose(fd);
        return -1;
    }
    fclose(fd);

    offset = 0;
    for(i=0; i<sub_file_size; i++){
        label = temp[offset++];
        label_offset = i*feeder->label_size + (int)label;
        feeder->binary_test_label[label_offset] = 1;

        for(j=0; j<feeder->num_pixels; j++){
            data_offset = i*feeder->num_pixels+j;
            feeder->binary_test_data[data_offset] = static_cast<float>(static_cast<unsigned char>(temp[offset++]));
        }
    }
    free(temp);

    /* Memory allocation for minibatch and labels. */
    feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
    feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);
    return 0;
}

static int pcnn_feeder_imagenet_init(struct feeder_t *feeder)
{
    int i, offset;
    int numEnts;
    char line[200];
    FILE *fd;

    /* settings from def.h */
    feeder->image_depth = IMAGENET_DEPTH;
    feeder->image_width = IMAGENET_WIDTH;
    feeder->image_height = IMAGENET_HEIGHT;
    feeder->image_orig_width = IMAGENET_WIDTH;
    feeder->image_orig_height = IMAGENET_HEIGHT;
    feeder->label_size = IMAGENET_LABEL_SIZE;
    sprintf(feeder->train_top_dir, IMAGENET_TRAIN_TOP_DIR);
    sprintf(feeder->test_top_dir, IMAGENET_TEST_TOP_DIR);
    sprintf(feeder->train_files, IMAGENET_TRAIN_LIST);
    sprintf(feeder->test_files, IMAGENET_TEST_LIST);

    feeder->num_pixels = feeder->image_depth * feeder->image_width * feeder->image_height;

    /* Training data */
    fd = fopen(feeder->train_files, "r");
    if(fd == NULL){
        printf("[%s][%d] %s fopen failed\n", __FUNCTION__,__LINE__, feeder->train_files);
        return -1;
    }

    numEnts = 0;
    while(fgets(line, 200, fd)!=NULL)
        numEnts++;

    feeder->num_train_images = numEnts;
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;

    feeder->train_offset = (int *)malloc(sizeof(int)*feeder->num_train_images);
    memset(feeder->train_offset, 0, sizeof(int)*feeder->num_train_images);
    fseek(fd, 0, SEEK_SET);

    i=0;
    offset=0;
    while(fgets(line, 200, fd)!=NULL){
        feeder->train_offset[i++] = offset;
        offset += strlen(line);
    }
    fclose(fd);

    feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
    feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);

    /* Validation data */
    fd = fopen(feeder->test_files, "r");
    if(fd == NULL){
        printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    numEnts = 0;
    while(fgets(line, 200, fd)!=NULL)
        numEnts++;
    feeder->num_test_images = numEnts;
    feeder->num_test_batches = feeder->num_test_images / feeder->batch_size;

    feeder->test_offset = (int *)malloc(sizeof(int)*feeder->num_test_images);
    memset(feeder->test_offset, 0, sizeof(int)*feeder->num_test_images);
    fseek(fd, 0, SEEK_SET);

    i=0;
    offset=0;
    while(fgets(line, 200, fd)!=NULL){
        feeder->test_offset[i++] = offset;
        offset += strlen(line);
    }
    fclose(fd);

    feeder->test_order = (int *)malloc(sizeof(int)*feeder->num_test_images);
    for(i=0; i<feeder->num_test_images; i++)
        feeder->test_order[i] = i;

    return 0;
}

static int pcnn_feeder_phantom_init(struct feeder_t *feeder)
{
    int i;
    int offset;
    char line[100];
    FILE *train_image_fd = NULL;
    FILE *test_image_fd = NULL;

    /* settings from def.h */
    feeder->image_depth = PHANTOM_DEPTH;
    feeder->image_width = PHANTOM_WIDTH;
    feeder->image_height = PHANTOM_HEIGHT;
    feeder->label_size = PHANTOM_LABEL_SIZE;
    feeder->image_orig_width = PHANTOM_ORIG_WIDTH;
    feeder->image_orig_height = PHANTOM_ORIG_HEIGHT;
    sprintf(feeder->train_files, PHANTOM_TRAIN_LIST);
    sprintf(feeder->test_files, PHANTOM_TEST_LIST);
    sprintf(feeder->train_top_dir, PHANTOM_TRAIN_TOP_DIR);
    sprintf(feeder->test_top_dir, PHANTOM_TEST_TOP_DIR);

    /* open list files */
	train_image_fd = fopen(feeder->train_files, "r");
	if(train_image_fd == NULL){
		printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
		return -1;
	}

	test_image_fd = fopen(feeder->test_files, "r");
	if(test_image_fd == NULL){
		printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
		return -1;
	}

    /* get the number of images and label images */
    feeder->num_train_images = 0;
	while(fgets(line, 100, train_image_fd)!=NULL){
		feeder->num_train_images++;
	}

    feeder->num_test_images = 0;
	while(fgets(line, 100, test_image_fd)!=NULL){
		feeder->num_test_images++;
	}

    feeder->num_pixels = feeder->image_depth * feeder->image_width * feeder->image_height;
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;
	feeder->num_test_batches = feeder->num_test_images / feeder->batch_size;

    /* memory allocation */
    if(feeder->do_crop){
        feeder->crop_offset_x = (int *)malloc(sizeof(int) * feeder->num_train_images);
        feeder->crop_offset_y = (int *)malloc(sizeof(int) * feeder->num_train_images);
    }

	feeder->test_order = (int *)malloc(sizeof(int)*feeder->num_test_images);
	for(i=0; i<feeder->num_test_images; i++)
		feeder->test_order[i] = i;

	feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
	feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);

    feeder->train_image_offset = (int *)malloc(sizeof(int)*feeder->num_train_images);
    feeder->test_image_offset = (int *)malloc(sizeof(int)*feeder->num_test_images);

    /* calculate offsets of the files in the list file */
    i=0;
	offset=0;
    fseek(train_image_fd, 0, SEEK_SET);
	while(fgets(line, 100, train_image_fd)!=NULL){
		feeder->train_image_offset[i++] = offset;
		offset += strlen(line);
	}

    i=0;
	offset=0;
    fseek(test_image_fd, 0, SEEK_SET);
	while(fgets(line, 100, test_image_fd)!=NULL){
		feeder->test_image_offset[i++] = offset;
		offset += strlen(line);
	}

    /* close list files */
    fclose(train_image_fd);
    fclose(test_image_fd);
    return 0;
}

static int pcnn_feeder_div2k_init(struct feeder_t *feeder)
{
    int i;
    int offset;
    char line[100];
    FILE *train_image_fd = NULL;
    FILE *test_image_fd = NULL;

    /* settings from def.h */
    feeder->image_depth = DIV2K_DEPTH;
    feeder->image_width = DIV2K_WIDTH;
    feeder->image_height = DIV2K_HEIGHT;
    feeder->label_size = DIV2K_LABEL_SIZE;
    sprintf(feeder->train_files, DIV2K_TRAIN_LIST);
    sprintf(feeder->test_files, DIV2K_TEST_LIST);
    sprintf(feeder->train_top_dir, DIV2K_TRAIN_TOP_DIR);
    sprintf(feeder->test_top_dir, DIV2K_TEST_TOP_DIR);

    /* open list files */
	train_image_fd = fopen(feeder->train_files, "r");
	if(train_image_fd == NULL){
		printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
		return -1;
	}

	test_image_fd = fopen(feeder->test_files, "r");
	if(test_image_fd == NULL){
		printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
		return -1;
	}

    /* get the number of images and label images */
    feeder->num_train_images = 0;
	while(fgets(line, 100, train_image_fd)!=NULL){
		feeder->num_train_images++;
	}

    feeder->num_test_images = 0;
	while(fgets(line, 100, test_image_fd)!=NULL){
		feeder->num_test_images++;
	}

    feeder->num_pixels = feeder->image_depth * feeder->image_width * feeder->image_height;
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;
	feeder->num_test_batches = feeder->num_test_images / feeder->batch_size;

    /* memory allocation */
    if(feeder->do_crop){
        feeder->crop_offset_x = (int *)malloc(sizeof(int) * feeder->num_train_images);
        feeder->crop_offset_y = (int *)malloc(sizeof(int) * feeder->num_train_images);
    }

	feeder->test_order = (int *)malloc(sizeof(int)*feeder->num_test_images);
	for(i=0; i<feeder->num_test_images; i++)
		feeder->test_order[i] = i;

	feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
	feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);

    feeder->train_image_offset = (int *)malloc(sizeof(int)*feeder->num_train_images);
    feeder->test_image_offset = (int *)malloc(sizeof(int)*feeder->num_test_images);

    /* calculate offsets of the files in the list file */
    i=0;
	offset=0;
    fseek(train_image_fd, 0, SEEK_SET);
	while(fgets(line, 100, train_image_fd)!=NULL){
		feeder->train_image_offset[i++] = offset;
		offset += strlen(line);
	}

    i=0;
	offset=0;
    fseek(test_image_fd, 0, SEEK_SET);
	while(fgets(line, 100, test_image_fd)!=NULL){
		feeder->test_image_offset[i++] = offset;
		offset += strlen(line);
	}

    /* close list files */
    fclose(train_image_fd);
    fclose(test_image_fd);
    return 0;
}

static int pcnn_feeder_ghost_init(struct feeder_t *feeder)
{
    int i;

    feeder->num_train_images = 2560;
    feeder->num_train_batches = feeder->num_train_images / feeder->batch_size;

    /* settings from def.h */
    feeder->image_depth = GHOST_BATCH_DEPTH;
    feeder->image_width = GHOST_BATCH_WIDTH;
    feeder->image_height = GHOST_BATCH_HEIGHT;
    feeder->image_orig_width = GHOST_BATCH_WIDTH;
    feeder->image_orig_height = GHOST_BATCH_HEIGHT;
    feeder->label_size = GHOST_BATCH_LABEL_SIZE;

    feeder->num_pixels = feeder->image_depth * feeder->image_width * feeder->image_height;

    feeder->test_order = (int *)malloc(sizeof(int)*feeder->num_test_images);
    for(i=0; i<feeder->num_test_images; i++)
        feeder->test_order[i] = i;

    /* Memory allocation for minibatch and labels. */
    feeder->minibatch = (float *)malloc(sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
    feeder->label = (float *)malloc(sizeof(float) * feeder->label_size * feeder->local_batch_size);
    return 0;
}

static int pcnn_feeder_read_meanfile(struct feeder_t *feeder)
{
    int i, j, ret=0, offset;
    size_t read_length=0;
    FILE *fd=NULL;
    char path[200];

    if(feeder->dataset == DATASET_MNIST){
        /* MNIST does not need preprocessing. */
    }
    else if(feeder->dataset == DATASET_CIFAR10){
        /* Load the mean image into the memory space. */
        feeder->mean_image = (float *)malloc(sizeof(float)*feeder->num_pixels);
        sprintf(path, "%s/mean_image.bin", CIFAR10_TRAIN_TOP_DIR);
        fd = fopen(path, "r");
        if(fd == NULL){
            printf("[%s][%d] opening %s failed.\n", __FUNCTION__, __LINE__, path);
            ret = -1;
        }
        else{
            read_length = fread(feeder->mean_image, sizeof(float), feeder->num_pixels, fd);
            if(read_length != (size_t)feeder->num_pixels){
                printf("[%s][%d] fread failed.\n", __FUNCTION__, __LINE__);
                ret = -1;
            }
            fclose(fd);
        }
    }
    else if(feeder->dataset == DATASET_IMAGENET){
        /* Load the mean pixels into the memory space. */
        feeder->mean_image = (float *)malloc(sizeof(float) * feeder->num_pixels);

        /* B */
        offset = 0;
        for(i=0; i<feeder->image_height; i++){
            for(j=0; j<feeder->image_width; j++){
                feeder->mean_image[offset++] = 103.936;
            }
        }

        /* G */
        for(i=0; i<feeder->image_height; i++){
            for(j=0; j<feeder->image_width; j++){
                feeder->mean_image[offset++] = 116.736;
            }
        }

        /* R */
        for(i=0; i<feeder->image_height; i++){
            for(j=0; j<feeder->image_width; j++){
                feeder->mean_image[offset++] = 124.16;
            }
        }
    }
    else if(feeder->dataset == DATASET_PHANTOM){
        feeder->mean_image = (float *)malloc(sizeof(float) * feeder->num_pixels);
        memset(feeder->mean_image, 0, sizeof(float) * feeder->num_pixels);
    }
    else if(feeder->dataset == DATASET_DIV2K){
        feeder->mean_image = (float *)malloc(sizeof(float) * feeder->num_pixels);
        memset(feeder->mean_image, 0, sizeof(float) * feeder->num_pixels);
    }
    else if(feeder->dataset == DATASET_GHOST){
        feeder->mean_image = (float *)malloc(sizeof(float) * feeder->num_pixels);
        memset(feeder->mean_image, 0, sizeof(float) * feeder->num_pixels);
    }
    else{
        printf("[%s][%d] invalid dataset flag!\n", __FUNCTION__, __LINE__);
        ret = -1;
    }

    return ret;
}

static int pcnn_feeder_get_batch_memory(int test, int batch_index, struct feeder_t *feeder, struct comm_queue_t *queue)
{
    int offset;
    int i, j,k,l;
    int start, end;
    int src_offset, frame_offset;
    int width_offset, height_offset;
    int large_area = (feeder->image_height + 8) * (feeder->image_width + 8);
    int batch_offset;
    int group_offset;
    float pixel_value;

    group_offset = queue->group_id * feeder->batch_size;
    batch_offset = queue->rank * feeder->local_batch_size;

    if(test == 0){/* Training set */
        if(feeder->dataset == DATASET_MNIST) {
            offset = feeder->train_order[batch_index + group_offset + batch_offset + i] * feeder->num_pixels;
            memcpy(feeder->minibatch, &feeder->binary_train_data[offset], sizeof(float) * feeder->local_batch_size * feeder->num_pixels);

            offset = feeder->train_order[batch_index + group_offset + batch_offset + i] * feeder->label_size;
            memcpy(feeder->label, &feeder->binary_train_label[offset], sizeof(float) * feeder->local_batch_size * feeder->label_size);
        }
        else{
            for(i=0; i<feeder->local_batch_size; i++){
                /* 1. Copy the original image into a large frame (height+8 x width+8). */
                src_offset = feeder->train_order[batch_index + group_offset + batch_offset + i] * feeder->num_pixels;
                for(j=0; j<feeder->image_depth; j++){
                    frame_offset = j * large_area + (feeder->image_width + 8) * 4;
                    for(k=0; k<feeder->image_height; k++){
                        for(l=0; l<feeder->image_width; l++){
                            feeder->large_frame[frame_offset + l + 4] = feeder->binary_train_data[src_offset + l];
                        }
                        src_offset += feeder->image_width;
                        frame_offset += (feeder->image_width + 8);
                    }
                }

                /* 2. Subtract the mean values. */
                src_offset = 0;
                for(j=0; j<feeder->image_depth; j++){
                    frame_offset = j * large_area + (feeder->image_width + 8) * 4;
                    for(k=0; k<feeder->image_height; k++){
                        for(l=0; l<feeder->image_width; l++){
                            pixel_value = static_cast<float>(static_cast<unsigned char>(feeder->large_frame[frame_offset + l + 4]));
                            feeder->large_frame[frame_offset + l + 4] = pixel_value - feeder->mean_image[src_offset + l];
                        }
                        src_offset += feeder->image_width;
                        frame_offset += (feeder->image_width + 8);
                    }
                }

                /* 3. Horizontally flip the image. */
                if(rand() % 2 == 1){
                    for(j=0; j<feeder->image_depth; j++){
                        for(k=0; k<feeder->image_height + 8; k++){
                            start = j * large_area + k * (feeder->image_width + 8);
                            end = start + feeder->image_width + 8;
                            while(start < end){
                                pixel_value = feeder->large_frame[start];
                                feeder->large_frame[start] = feeder->large_frame[end];
                                feeder->large_frame[end] = pixel_value;
                                start++;
                                end--;
                            }
                        }
                    }
                }

                /* 4. Extract a random (height x width) patch from the frame and move it to minibatch buffer. */
                height_offset = rand() % 8;
                width_offset = rand() % 8;

                offset = i * feeder->num_pixels;
                for(j=0; j<feeder->image_depth; j++){
                    for(k=0; k<feeder->image_height; k++){
                        for(l=0; l<feeder->image_width; l++){
                            src_offset = (j * (feeder->image_width + 8) * (feeder->image_height + 8)) +
                                         ((k + height_offset) * (feeder->image_width + 8)) +
                                         (l + width_offset);
                            feeder->minibatch[offset++] = feeder->large_frame[src_offset];
                        }
                    }
                }

                /* Read labels. */
                offset = feeder->train_order[batch_index + group_offset + batch_offset + i];
                memcpy(&feeder->label[i * feeder->label_size],
                       &feeder->binary_train_label[offset * feeder->label_size],
                       sizeof(float) * feeder->label_size);
            }
        }
    }
    else{/* Test set */
        offset = (batch_index + group_offset + batch_offset) * feeder->num_pixels;
        memcpy(feeder->minibatch, &feeder->binary_test_data[offset], sizeof(float) * feeder->local_batch_size * feeder->num_pixels);

        offset = (batch_index + group_offset + batch_offset) * feeder->label_size;
        memcpy(feeder->label, &feeder->binary_test_label[offset], sizeof(float) * feeder->local_batch_size * feeder->label_size);
    }
    return 0;
}

static int pcnn_feeder_read_opencv(int test, char **paths, struct feeder_t *feeder)
{
    int i,j,k,S;
    int offset_r, offset_g, offset_b;
    int row_offset,col_offset;
    char line[500];
    double scale;
    Mat img;
    Mat resized_img;
    Mat cropped_img;
    Vec3b pixel;

    for(i=0; i<feeder->local_batch_size; i++){
        if(test == 0)
            sprintf(line, "%s/%s", feeder->train_top_dir, paths[i]);
        else
            sprintf(line, "%s/%s", feeder->test_top_dir, paths[i]);

        img = imread(line);

        /* Resize the image first. */
        S = (img.rows < img.cols)?img.rows:img.cols;
        scale = (double)256 / (double)S;
        resize(img, resized_img, Size(scale*img.cols, scale*img.rows), 0, 0, CV_INTER_NN);

        /* Then, crop it to a 224 x 224 patch. */
        row_offset = rand()%(resized_img.rows - feeder->image_height);
        col_offset = rand()%(resized_img.cols - feeder->image_width);
        cropped_img = resized_img(Rect(col_offset,
                                       row_offset,
                                       feeder->image_width,
                                       feeder->image_height)).clone();

        offset_b = i*feeder->num_pixels;
        offset_g = (cropped_img.rows * cropped_img.cols) + offset_b;
        offset_r = (cropped_img.rows * cropped_img.cols) + offset_g;
        for(j=0; j<cropped_img.rows; j++){
            for(k=0; k<cropped_img.cols; k++){
                pixel = cropped_img.at<Vec3b>(Point(k,j));
                feeder->minibatch[offset_b++] = pixel[0];
                if(cropped_img.channels() < 3){
                    feeder->minibatch[offset_g++] = pixel[0];
                    feeder->minibatch[offset_r++] = pixel[0];
                }
                else{
                    feeder->minibatch[offset_g++] = pixel[1];
                    feeder->minibatch[offset_r++] = pixel[2];
                }
            }
        }
    }
    return 0;
}

static int pcnn_feeder_get_batch_imagenet(int test, int batch_index,
                                          struct feeder_t *feeder,
                                          struct comm_queue_t *queue)
{
    int i,image,label;
    int batch_offset;
    int group_offset;
    FILE *fd;
    char *temp;
    char **paths;
    char line[500];

    paths = (char **)malloc(sizeof(char *) * feeder->local_batch_size);
    for(i=0; i<feeder->local_batch_size; i++){
        paths[i] = (char *)malloc(sizeof(char)*200);
        memset(paths[i], 0, sizeof(char)*200);
    }
    memset(feeder->label, 0, sizeof(float) * feeder->label_size * feeder->local_batch_size);
    memset(feeder->minibatch, 0, sizeof(float) * feeder->num_pixels * feeder->local_batch_size);

    /* Get the file names and labels first. */
    if(test == 0)
        fd = fopen(feeder->train_files, "r");
    else
        fd = fopen(feeder->test_files, "r");
    if(fd == NULL){
        printf("[%s][%d] fopen failed\n", __FUNCTION__,__LINE__);
        return -1;
    }

    /* Given a file that contains the training file names,
     * create a list of randomly chosen file names (paths).
     * Then, pcnn_feeder_read_opencv() actually reads the selected images. */
    group_offset = queue->group_id * feeder->batch_size;
    batch_offset = queue->rank * feeder->local_batch_size;
    for(i=0; i<feeder->local_batch_size; i++){
        if(test == 0){
            image = feeder->train_order[batch_index + group_offset + batch_offset + i];
            fseek(fd, feeder->train_offset[image], SEEK_SET);
        }
        else{
            image = feeder->test_order[batch_index + group_offset + batch_offset + i];
            fseek(fd, feeder->test_offset[image], SEEK_SET);
        }

        if((fgets(line, 500, fd))!= NULL){
            temp = strtok(line, " ");
            if(temp == NULL){
                printf("[%s][%d] list file is invalid.\n", __FUNCTION__,__LINE__);
                return -1;
            }
            memcpy(paths[i], temp, sizeof(char)*strlen(temp));
            label = atoi(strtok(NULL, "\n"));
            feeder->label[i*feeder->label_size+label] = 1;
        }
    }
    fclose(fd);

    /* Then read the actual images. */
    pcnn_feeder_read_opencv(test, paths, feeder);

    for(i=0; i<feeder->local_batch_size; i++)
        free(paths[i]);
    free(paths);
    return 0;
}

static int pcnn_feeder_read_jpeg(int test, int image_index, char **image_paths, char **label_paths, struct feeder_t *feeder)
{
    int i,j,k;
    int offset;
    char line[500];
    Mat img;
    Mat cropped_img;
    Vec3b pixel;

    /* Read images first. */
    for(i=0; i<feeder->batch_size; i++){
        if(test == 0)
            sprintf(line, "%s/LR/%s", feeder->train_top_dir, image_paths[i]);
        else
            sprintf(line, "%s/LR/%s", feeder->test_top_dir, image_paths[i]);
        img = imread(line, CV_LOAD_IMAGE_COLOR);

        /* crop the image */
        if(feeder->do_crop == 1){
            cropped_img = img(Rect(feeder->crop_offset_x[i + image_index],
                                   feeder->crop_offset_y[i + image_index],
                                   feeder->image_width,
                                   feeder->image_height)).clone();

            /* load the image into minibatch buffer */
            offset = i*feeder->num_pixels;
            for(j=0; j<cropped_img.rows; j++){
                for(k=0; k<cropped_img.cols; k++){
                    pixel = cropped_img.at<Vec3b>(Point(k,j));
                    feeder->minibatch[offset++] = pixel[0];
                }
            }
        }
        else{
            /* load the image into minibatch buffer */
            offset = i*feeder->num_pixels;
            for(j=0; j<img.rows; j++){
                for(k=0; k<img.cols; k++){
                    pixel = img.at<Vec3b>(Point(k,j));
                    feeder->minibatch[offset++] = pixel[0];
                }
            }
        }
    }

    /* Read labels then. */
    for(i=0; i<feeder->batch_size; i++){
        if(test == 0)
            sprintf(line, "%s/HR/%s", feeder->train_top_dir, label_paths[i]);
        else
            sprintf(line, "%s/HR/%s", feeder->test_top_dir, label_paths[i]);
        img = imread(line, CV_LOAD_IMAGE_COLOR);

        /* crop the image */
        if(feeder->do_crop == 1){
            cropped_img = img(Rect(feeder->crop_offset_x[i + image_index],
                                   feeder->crop_offset_y[i + image_index],
                                   feeder->image_width,
                                   feeder->image_height)).clone();

            /* load the label image into label buffer */
            offset = i*feeder->num_pixels;
            for(j=0; j<cropped_img.rows; j++){
                for(k=0; k<cropped_img.cols; k++){
                    pixel = cropped_img.at<Vec3b>(Point(k,j));
                    feeder->label[offset++] = pixel[0];
                }
            }
        }
        else{
            /* load the label image into label buffer */
            offset = i*feeder->num_pixels;
            for(j=0; j<img.rows; j++){
                for(k=0; k<img.cols; k++){
                    pixel = img.at<Vec3b>(Point(k,j));
                    feeder->label[offset++] = pixel[0];
                }
            }
        }
    }
    return 0;
}

static int pcnn_feeder_read_and_crop(int test, int upsample_ratio, char **image_paths, char **label_paths, struct feeder_t *feeder)
{
    int i,j,k;
    int crop_row_off;
    int crop_col_off;
    int num_row_blocks;
    int num_col_blocks;
    int offset_b, offset_g, offset_r;
    char line[500];
    char *name;
    Mat img;
    Mat cropped_img;
    Vec3b pixel;

    for(i=0; i<feeder->local_batch_size; i++){
        /* Read images first. */
        name = strtok(image_paths[i], ".");
        if(test == 0)
            sprintf(line, "%s/LR/X2/%sx2.png", feeder->train_top_dir, name);
        else
            sprintf(line, "%s/LR/X2/%sx2.png", feeder->test_top_dir, name);
        img = imread(line, CV_LOAD_IMAGE_COLOR);

        /* Extract a random patch from each image. */
        if(test == 0){
            crop_row_off = (rand() % (img.rows - feeder->image_height + 1));
            crop_col_off = (rand() % (img.cols - feeder->image_width + 1));
        }
        else{
            num_row_blocks = img.rows / feeder->image_height;
            num_col_blocks = img.cols / feeder->image_width;
            
            crop_row_off = num_row_blocks / 2 * feeder->image_height;
            crop_col_off = num_col_blocks / 2 * feeder->image_width;
        }
        cropped_img = img(Rect(crop_col_off, crop_row_off, feeder->image_width, feeder->image_height)).clone();

        /* load the image into minibatch buffer */
        offset_b = i*feeder->num_pixels;
        offset_g = (cropped_img.rows * cropped_img.cols) + offset_b;
        offset_r = (cropped_img.rows * cropped_img.cols) + offset_g;
        for(j=0; j<cropped_img.rows; j++){
            for(k=0; k<cropped_img.cols; k++){
                pixel = cropped_img.at<Vec3b>(Point(k,j));
                feeder->minibatch[offset_b++] = pixel[0];
                if(cropped_img.channels() < 3){
                    feeder->minibatch[offset_g++] = pixel[0];
                    feeder->minibatch[offset_r++] = pixel[0];
                }
                else{
                    feeder->minibatch[offset_g++] = pixel[1];
                    feeder->minibatch[offset_r++] = pixel[2];
                }
            }
        }

        /* Read labels then. */
        name = strtok(image_paths[i], ".");
        if(test == 0)
            sprintf(line, "%s/HR/%s.png", feeder->train_top_dir, name);
        else
            sprintf(line, "%s/HR/%s.png", feeder->test_top_dir, name);
        img = imread(line, CV_LOAD_IMAGE_COLOR);

        /* Extract a random patch from each image. */
        crop_row_off *= upsample_ratio;
        crop_col_off *= upsample_ratio;
        cropped_img = img(Rect(crop_col_off, crop_row_off, feeder->image_width * upsample_ratio, feeder->image_height * upsample_ratio)).clone();

        /* load the image into minibatch buffer */
        offset_b = i*feeder->label_size;
        offset_g = (cropped_img.rows * cropped_img.cols) + offset_b;
        offset_r = (cropped_img.rows * cropped_img.cols) + offset_g;
        for(j=0; j<cropped_img.rows; j++){
            for(k=0; k<cropped_img.cols; k++){
                pixel = cropped_img.at<Vec3b>(Point(k,j));
                feeder->label[offset_b++] = pixel[0];
                if(cropped_img.channels() < 3){
                    feeder->label[offset_g++] = pixel[0];
                    feeder->label[offset_r++] = pixel[0];
                }
                else{
                    feeder->label[offset_g++] = pixel[1];
                    feeder->label[offset_r++] = pixel[2];
                }
            }
        }
    }
    return 0;
}

static int pcnn_feeder_get_batch_phantom(int test, int batch_index, struct feeder_t *feeder)
{
	int i,image;
	FILE *image_fd=NULL;
    int *order;
	char *temp;
	char **image_paths;
	char **label_paths;
	char line[500];

	image_paths = (char **)malloc(sizeof(char *)*feeder->batch_size);
	label_paths = (char **)malloc(sizeof(char *)*feeder->batch_size);
	for(i=0; i<feeder->batch_size; i++){
		image_paths[i] = (char *)malloc(sizeof(char)*200);
		memset(image_paths[i], 0, sizeof(char)*200);

		label_paths[i] = (char *)malloc(sizeof(char)*200);
		memset(label_paths[i], 0, sizeof(char)*200);
	}
	memset(feeder->label, 0, sizeof(float)*feeder->label_size*feeder->batch_size);
	memset(feeder->minibatch, 0, sizeof(float)*feeder->num_pixels*feeder->batch_size);

    /* 1. open the files */
    if(test == 0)
        image_fd = fopen(feeder->train_files, "r");
    else
        image_fd = fopen(feeder->test_files, "r");
    if(image_fd == NULL){
        printf("[%s][%d] fopen failed\n", __FUNCTION__, __LINE__);
        return -1;
    }

    order = test ? feeder->test_order : feeder->train_order;
    for(i=0; i<feeder->batch_size; i++){
        if(test == 0){
            image = order[batch_index + i];
            fseek(image_fd, feeder->train_image_offset[image], SEEK_SET);
        }
        else{
            image = order[batch_index + i];
            fseek(image_fd, feeder->test_image_offset[image], SEEK_SET);
        }

		if((fgets(line, 500, image_fd))!= NULL){
			temp = strtok(line, "\n");
			if(temp == NULL){
				printf("[%s][%d] list file is invalid.\n", __FUNCTION__,__LINE__);
				return -1;
			}
			memcpy(image_paths[i], temp, sizeof(char)*strlen(temp));
			memcpy(label_paths[i], temp, sizeof(char)*strlen(temp));
		}
    }

    fclose(image_fd);

    /* read the actual images and labels */
	pcnn_feeder_read_jpeg(test, batch_index, image_paths, label_paths, feeder);

	for(i=0; i<feeder->batch_size; i++){
		free(image_paths[i]);
		free(label_paths[i]);
	}
	free(image_paths);
	free(label_paths);

	return 0;
}

static int pcnn_feeder_get_batch_div2k(int test, int batch_index, int upsample_ratio, struct feeder_t *feeder, struct comm_queue_t *queue)
{
	int i, image;
    int batch_offset;
    int group_offset;
	FILE *image_fd=NULL;
    int *order;
	char *temp;
	char **image_paths;
	char **label_paths;
	char line[500];

	image_paths = (char **)malloc(sizeof(char *) * feeder->local_batch_size);
	label_paths = (char **)malloc(sizeof(char *) * feeder->local_batch_size);
	for(i=0; i<feeder->local_batch_size; i++){
		image_paths[i] = (char *)malloc(sizeof(char)*200);
		memset(image_paths[i], 0, sizeof(char)*200);

		label_paths[i] = (char *)malloc(sizeof(char)*200);
		memset(label_paths[i], 0, sizeof(char)*200);
	}
	memset(feeder->label, 0, sizeof(float) * feeder->label_size * feeder->local_batch_size);
	memset(feeder->minibatch, 0, sizeof(float) * feeder->num_pixels * feeder->local_batch_size);

    /* 1. open the files */
    if(test == 0)
        image_fd = fopen(feeder->train_files, "r");
    else
        image_fd = fopen(feeder->test_files, "r");
    if(image_fd == NULL){
        printf("[%s][%d] fopen failed\n", __FUNCTION__, __LINE__);
        return -1;
    }

    group_offset = queue->group_id * feeder->batch_size;
    batch_offset = queue->rank * feeder->local_batch_size;
    order = test ? feeder->test_order : feeder->train_order;
    for(i=0; i<feeder->local_batch_size; i++){
        image = order[batch_index + group_offset + batch_offset + i];
        if(test == 0)
            fseek(image_fd, feeder->train_image_offset[image], SEEK_SET);
        else
            fseek(image_fd, feeder->test_image_offset[image], SEEK_SET);

		if((fgets(line, 500, image_fd))!= NULL){
			temp = strtok(line, "\n");
			if(temp == NULL){
				printf("[%s][%d] list file is invalid.\n", __FUNCTION__,__LINE__);
				return -1;
			}
			memcpy(image_paths[i], temp, sizeof(char)*strlen(temp));
			memcpy(label_paths[i], temp, sizeof(char)*strlen(temp));
		}
    }
    fclose(image_fd);

    /* read the actual images and labels */
	pcnn_feeder_read_and_crop(test, upsample_ratio, image_paths, label_paths, feeder);

	for(i=0; i<feeder->local_batch_size; i++){
		free(image_paths[i]);
		free(label_paths[i]);
	}
	free(image_paths);
	free(label_paths);

	return 0;
}

static int pcnn_feeder_get_ghost_batch(int test, int batch_index, struct feeder_t *feeder)
{
	memset(feeder->label, 0.1, sizeof(float) * feeder->label_size * feeder->local_batch_size);
	memset(feeder->minibatch, 0.1, sizeof(float) * feeder->num_pixels * feeder->local_batch_size);
	return 0;
}

void pcnn_feeder_generate_crop_offsets(struct feeder_t *feeder)
{
    int i;
    if(feeder->do_crop == 0)
        return;

    for(i=0; i<feeder->num_train_images; i++){
        feeder->crop_offset_x[i] = (rand() % (feeder->image_orig_width - feeder->image_width + 1));
        feeder->crop_offset_y[i] = (rand() % (feeder->image_orig_height - feeder->image_height + 1));
    }
}
