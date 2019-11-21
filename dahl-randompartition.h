#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  const void *sexp_ptr;
  int32_t *data_ptr;
  int32_t len;
} RR_SEXP_vector_INTSXP;

extern double callRFunction_logIntegratedLikelihoodOfSubset(const void *fn_ptr,
                                                            RR_SEXP_vector_INTSXP indices,
                                                            const void *env_ptr);

extern double callRFunction_logPosteriorPredictiveOfItem(const void *fn_ptr,
                                                         int32_t i,
                                                         RR_SEXP_vector_INTSXP indices,
                                                         const void *env_ptr);

void dahl_randompartition__crp_partition(int32_t do_sampling,
                                         int32_t n_partitions,
                                         int32_t n_items,
                                         int32_t *partition_labels_ptr,
                                         double *partition_probs_ptr,
                                         const int32_t *seed_ptr,
                                         double mass);

void dahl_randompartition__epa_partition(int32_t do_sampling,
                                         int32_t n_partitions,
                                         int32_t n_items,
                                         int32_t *partition_labels_ptr,
                                         double *partition_probs_ptr,
                                         const int32_t *seed_ptr,
                                         double *similarity_ptr,
                                         const int32_t *permutation_ptr,
                                         double mass,
                                         double discount,
                                         int32_t use_random_permutation);

void dahl_randompartition__focal_partition(int32_t n_partitions,
                                           int32_t n_items,
                                           const int32_t *focal_ptr,
                                           const double *weights_ptr,
                                           const int32_t *permutation_ptr,
                                           double mass,
                                           int32_t do_sampling,
                                           int32_t use_random_permutations,
                                           int32_t *partition_labels_ptr,
                                           double *partition_probs_ptr,
                                           const int32_t *seed_ptr);

void dahl_randompartition__mhrw_update(int32_t n_attempts,
                                       int32_t n_items,
                                       double rate,
                                       double mass,
                                       int32_t *partition_ptr,
                                       const void *log_likelihood_function_ptr,
                                       const void *env_ptr,
                                       int32_t *n_accepts,
                                       const int32_t *seed_ptr);

void dahl_randompartition__neal_algorithm3_crp(int32_t n_updates_for_partition,
                                               int32_t n_items,
                                               int32_t *partition_ptr,
                                               int32_t prior_only,
                                               const void *log_posterior_predictive_function_ptr,
                                               const void *env_ptr,
                                               const int32_t *seed_ptr,
                                               double mass);

void dahl_randompartition__neal_algorithm3_nggp(int32_t n_updates_for_partition,
                                                int32_t n_items,
                                                int32_t *partition_ptr,
                                                int32_t prior_only,
                                                const void *log_posterior_predictive_function_ptr,
                                                const void *env_ptr,
                                                const int32_t *seed_ptr,
                                                int32_t n_updates_for_u,
                                                double *u_ptr,
                                                double mass,
                                                double reinforcement);

void dahl_randompartition__nggp_partition(int32_t do_sampling,
                                          int32_t n_partitions,
                                          int32_t n_items,
                                          int32_t *partition_labels_ptr,
                                          double *partition_probs_ptr,
                                          const int32_t *seed_ptr,
                                          double u,
                                          double mass,
                                          double reinforcement,
                                          int32_t n_updates_for_u);

extern RR_SEXP_vector_INTSXP rrAllocVectorINTSXP(int32_t len);
