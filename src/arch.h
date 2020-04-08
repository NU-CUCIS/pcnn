/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
void pcnn_arch_config_lenet(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_cifar10(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_vgga(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_resnet20(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_resnet50(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_edsr(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_drrn(struct model_t *model, struct feeder_t *feeder);
void pcnn_arch_config_vdsr(struct model_t *model, struct feeder_t *feeder);
