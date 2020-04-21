/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_loss_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder);
void pcnn_loss_bp(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
