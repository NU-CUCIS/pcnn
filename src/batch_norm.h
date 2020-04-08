/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_bn_ff(int op, struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
void pcnn_bn_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
void pcnn_bn_update(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder, struct comm_queue_t *queue);
