#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

extern double callRFunction_indices_f64(const void *fn_ptr,
                                        const void *indices_ptr,
                                        int32_t len,
                                        const void *env_ptr);

void dahl_randompartition__crp__sample(int32_t n_partitions,
                                       int32_t n_items,
                                       double mass,
                                       int32_t *ptr);

void dahl_randompartition__focal_partition(int32_t n_partitions,
                                           int32_t n_items,
                                           const int32_t *focal_ptr,
                                           const double *weights_ptr,
                                           const int32_t *permutation_ptr,
                                           double mass,
                                           int32_t do_sampling,
                                           int32_t use_random_permutations,
                                           int32_t *partition_labels_ptr,
                                           double *partition_probs_ptr);

void dahl_randompartition__mhrw_update(int32_t n_updates,
                                       int32_t n_items,
                                       double rate,
                                       double mass,
                                       int32_t *partition_ptr,
                                       const void *log_likelihood_function_ptr,
                                       const void *env_ptr,
                                       int32_t *n_accepts);
