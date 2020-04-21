/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <pthread.h>
#include <mpi.h>
#include <omp.h>

struct feature_t{
    int type;
    int sub_type;

    int ReLU;
    int loss_type;
    int batch_norm;

    int output_channels;
    int filter_depth;
    int filter_rows;
    int filter_cols;

    int num_channels;
    int image_depth;
    int image_rows;
    int image_cols;

    int pad_depth;
    int pad_rows;
    int pad_cols;
    int stride_depth;
    int stride_rows;
    int stride_cols;

    float mean;
    float std;

    int bottom_layer;
    int skip_from;
    float res_scale;
};

struct layer_t{
    int type; /* Conv./Pool./Full. */
    int sub_type; /* 0: max-pooling 1: avg-pooling */
    int id; 
    int bottom_trainable_layer;
    int top_trainable_layer;
    int bottom_layer;

    int input_channels;
    int input_depth;
    int input_rows;
    int input_cols;

    int filter_depth;
    int filter_rows;
    int filter_cols;

    int output_channels;
    int output_depth;
    int output_rows;
    int output_cols;

	float mean;
	float std;

    int filter_size;
	int bias_size;

    int local_weight_count;
    int local_weight_off;

    int num_neurons;
    int num_prev_neurons;

    /* If the filter size is not divisible by the number of processes,
     * We use MPI_Alltoallv and MPI_Allgatherv for communications. */
    int aligned_weight;

    /* pointers */
    int *sdispls_weight;
    int *rdispls_weight;
    int *scounts_weight;
    int *rcounts_weight;

    int num_local_gradients;
    int num_gradients;
    int aligned_gradients;
    int *sdispls_gradients;
    int *rdispls_gradients;
    int *scounts_gradients;
    int *rcounts_gradients;

    float *weight;
    float *bias;

    int pad_depth;
    int pad_rows;
    int pad_cols;
    int stride_depth;
    int stride_rows;
    int stride_cols;

    int ReLU;
    int loss_type;
    int batch_norm;

    float *a;
    float *e;
    float *recv_a;
    float *recv_e;
    float *rep_a;
    float *rep_e;

    int *poolmap;

    float *local_sumws;
    float *local_sumbs;
    float *global_sumws;
    float *global_sumbs;
    float *prev_sumws;
    float *prev_sumbs;
    float *m_sumws;
    float *m_sumbs;
    float *v_sumws;
    float *v_sumbs;

    /* batch normalization data */
    float *gamma;
    float *beta;
    float *a_norm;
    float *sqrt_var;
    float *global_mean;
    float *global_variance;
    float *local_dgamma;
    float *global_dgamma;
    float *local_dbeta;
    float *global_dbeta;
    float *prev_dgamma;
    float *prev_dbeta;
    float *m_dgamma;
    float *m_dbeta;
    float *v_dgamma;
    float *v_dbeta;
    float bn_scale_factor;

    /* residual connection */
    int skip_from;
    float res_scale;
};

struct param_t{
    float *params;
    float *gradients;
    float *gradient_sums;
    float *prev_gradient_sums;
    float *m_gradient_sums;
    float *v_gradient_sums;
    float *bn_params;
    float *bn_global_statistics;
    float *bn_gradients;
    float *bn_gradient_sums;
    float *bn_prev_gradients;
    float *bn_m_gradients;
    float *bn_v_gradients;
    float *col;
    float *pool2full;

    float *local_conv_grads;
    float *global_conv_grads;
    float *local_full_grads;
    float *global_full_grads;

    float *prev_conv_grads;
    float *prev_full_grads;

    float local_loss;
    float global_loss;
    float epoch_loss;

    /* batch normalization */
    float *multiplier;
    float *sums;

    /* Adam */
    float beta1_decay;
    float beta2_decay;

    /* flags */
    int epoch;
    int current_index;
    int current_test_index;
    int num_updates;
    int num_processed_batches;
    int num_trained_epochs;
    int total_size;
    int bn_param_size;
    int bn_global_statistics_size;
    int bn_num_layers;
    int conv_weight_size;
    int conv_bias_size;
    int conv_total_size;
    int full_weight_size;
    int full_bias_size;
    int full_total_size;
    int conv_grads_size;
    int full_grads_size;
    int first_full_id;

    /* output metrics */
    float custom_output;
    int num_corrects;
};

struct model_t{
    int mode; // mode 0: training / model 1: evaluating 
    int task_type;
    int param_init_method;
    int num_layers;
    int num_epochs;

    float loss;
    float learning_rate;
    float decay_factor;
    int decay_steps;

    float upsample_ratio;

    /* SGD */
    float momentum;
    float weight_decay;

    /* Adam */
    float beta1;
    float beta2;
    float epsilon;

    /* batch normalization */
    float eps;
    float moving_average_fraction;

    struct layer_t **layers;

    pthread_t comm_thread;
    pthread_t comp_thread;

    /* flags */
    int optimizer;
    int overlap;
    int comm_pattern;

    /* function pointers for feedforward and backpropagation stages */
    void (*feedforward)(int, struct feeder_t *, struct model_t *, struct param_t *, struct comm_queue_t *);
    void (*backprop)(int, struct feeder_t *, struct model_t *, struct param_t *, struct comm_queue_t *);
    void (*update)(struct model_t *, struct param_t *param, struct feeder_t *, struct comm_queue_t *);

    /* statistics */
    unsigned long param_size;
    unsigned long intermediate_size;

    /* metadata */
    int test_per_epoch;
    int checkpoint_interval;
    char *checkpoint_path;
};

struct msg_t{
    struct model_t *model;
    struct param_t *param;
    struct feeder_t *feeder;
    struct comm_queue_t *queue; // for each comm thread
};

struct model_t *pcnn_model_init(int num_train_images, int num_epochs, int mode, struct comm_queue_t *queue);
struct param_t *pcnn_model_init_param(struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_model_destroy(struct model_t *model);
void pcnn_model_free_param(struct model_t *model, struct param_t *param);
void pcnn_model_init_layer(struct model_t *model, struct feeder_t *feeder, struct feature_t *features);
void pcnn_model_update_layer(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_model_partial_update_conv_layer(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_model_partial_update_full_layer(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_model_get_default_features(struct feature_t *features);
void pcnn_model_decay_learning_rate(struct model_t *model, struct param_t *param);
void pcnn_model_init_comm_offsets(struct model_t *model, struct comm_queue_t *queue);
void pcnn_model_put_momentum_together(struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
void pcnn_model_update_interval_layer(int id, struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
void pcnn_model_update_interval_model(struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
