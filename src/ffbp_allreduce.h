/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_ffbp_allreduce_feedforward(int op,
                                     struct feeder_t *feeder,
                                     struct model_t *model,
                                     struct param_t *param,
                                     struct comm_queue_t *queue);
void pcnn_ffbp_allreduce_backprop(int op,
                                  struct feeder_t *feeder,
                                  struct model_t * model,
                                  struct param_t *param,
                                  struct comm_queue_t *queue);
void pcnn_ffbp_allreduce_update(struct model_t *model,
                                struct param_t *param,
                                struct feeder_t *feeder,
                                struct comm_queue_t *queue);
