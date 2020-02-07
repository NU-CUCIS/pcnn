/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_upsample_ff(int count, int ratio, struct layer_t *bottom, struct layer_t *top);
void pcnn_upsample_bp(int count, int ratio, struct layer_t *bottom, struct layer_t *top);
