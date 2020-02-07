/*
 * Copyright (C) 2020, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 */
struct comm_req_t{
    int type;
    int layer_id;
};

struct comm_queue_t{
    struct comm_req_t *queue;
    int queue_size;
    int req_count;
    int index;

    /* Flags */
    int flag_reduce;
    int flag_reduce_l;
    int flag_reduce_ag;
    int *flag_reduce_g;
    int *flag_reduce_p;
    int *flag_gather_g;
    int *flag_gather_e;
    int *flag_gather_w;
    int *flag_gather_m0;
    int *flag_gather_m1;
    int *flag_all2all_a;;
    int *flag_all2all_g;;

    pthread_cond_t cond;
    pthread_mutex_t mut;

    int num_groups;
    int rank;
    int nproc;
    int group_id;
    int sync_interval;
    MPI_Comm comm;
    MPI_Comm across;
    MPI_Comm world;
};

struct comm_queue_t *pcnn_comm_init(int num_groups, int sync_interval);
void pcnn_comm_destroy(struct comm_queue_t *queue);
int pcnn_comm_insert_req(struct model_t *model, struct comm_queue_t *queue, struct comm_req_t *req);
void *pcnn_comm_thread(void *ptr);
