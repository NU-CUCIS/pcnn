/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
int pcnn_record_checkpoint(struct param_t *param, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue);
int pcnn_record_continue_training(char *bin_path, struct param_t *param, struct model_t *model, struct comm_queue_t *queue);

/* Debugging-purposed functions */
int pcnn_record_gradients(int r, int dpidx, struct model_t *model, struct param_t *param, struct comm_queue_t *queue);
int pcnn_record_param(struct model_t *model, struct param_t *param, int r);
int pcnn_record_transfer_learning_txt(struct model_t *model, struct param_t *param, char *bin_path);
