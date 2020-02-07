/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_pool_ff(int op, int count, struct layer_t *bottom, struct layer_t *top, struct param_t *param);
void pcnn_pool_bp(int count, struct layer_t *bottom, struct layer_t *top);
