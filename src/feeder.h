/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
struct feeder_t{
    int dataset;
    int num_train_images;
    int num_train_batches;
    int num_test_images;
    int num_test_batches;
    int image_channels;
    int image_depth;
    int image_width;
    int image_height;
    int image_orig_width;
    int image_orig_height;
    int num_pixels;
    int label_size;
    int batch_size;
    int local_batch_size;

    int do_shuffle;
    int do_crop;
    char train_top_dir[200];
    char test_top_dir[200];
    char train_files[200];
    char test_files[200];

    int *crop_offset_x;
    int *crop_offset_y;

    int *train_order;
    int *train_offset;
    int *train_image_offset;
    int *test_image_offset;
    int *test_order;
    int *test_offset;
    float *minibatch;
    float *label;
    float *mean_image;
    float *binary_train_data;
    float *binary_train_label;
    float *binary_test_data;
    float *binary_test_label;
    float *large_frame;
};

struct feeder_t *pcnn_feeder_init(int do_shuffle, struct comm_queue_t *queue);
void pcnn_feeder_shuffle(struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_feeder_mean_subtract(struct feeder_t *feeder);
void pcnn_feeder_subtract_mean_image(struct feeder_t *feeder);
int pcnn_feeder_get_minibatch(int test, int batch_index, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_feeder_destroy(struct feeder_t *feeder);
void pcnn_feeder_generate_crop_offsets(struct feeder_t *feeder);
