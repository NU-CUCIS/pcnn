/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
int pcnn_full_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param);
int pcnn_full_bp(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param);
void pcnn_full_gradw_pattern0(struct layer_t *bottom, struct layer_t *top, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
void pcnn_full_gradb_pattern0(struct layer_t *top, struct model_t *model, struct feeder_t *feeder);
void pcnn_full_gradw_pattern1(struct layer_t *bottom, struct layer_t *top, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_full_gradb_pattern1(struct layer_t *top, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue);
