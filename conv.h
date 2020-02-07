/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
int pcnn_conv_ff(int op, int imgidx, int count, 
				struct layer_t *bottom, struct layer_t *top, 
				struct feeder_t *feeder, struct param_t *param);

int pcnn_conv_bp(int op, int count,
				struct layer_t *bottom, struct layer_t *top,
				struct param_t *param);

void pcnn_conv_gradw(int op, int imgidx, int count, 
				struct layer_t *bottom, struct layer_t *top, 
				struct feeder_t *feeder, struct param_t *param);

void pcnn_conv_gradb(struct layer_t *layer, struct model_t *model, struct param_t *param, struct feeder_t *feeder);
