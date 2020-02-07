/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_relu_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder);
void pcnn_relu_bp(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder);
