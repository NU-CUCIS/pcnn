/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_residual_ff(struct layer_t *right, struct model_t *model, struct feeder_t *feeder, struct comm_queue_t *queue);
void pcnn_residual_bp(struct layer_t *right, struct model_t *model, struct feeder_t *feeder);
