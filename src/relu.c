/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include "def.h"
#include "model.h"
#include "feeder.h"

void pcnn_relu_ff(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder)
{
    int i = 0;
    const int length = feeder->local_batch_size * layer->num_neurons;

#pragma omp parallel for
    for(i=0; i<length; i++)
        layer->a[i] = layer->a[i] > 0.0f ? layer->a[i] : 0.0f;
}

void pcnn_relu_bp(struct layer_t *layer, struct model_t *model, struct feeder_t *feeder)
{
    int i = 0;
    const int length = feeder->local_batch_size * layer->num_neurons;

#pragma omp parallel for
    for(i=0; i<length; i++)
        layer->e[i] = layer->a[i] == 0.0f ? 0.0f : layer->e[i];
}
