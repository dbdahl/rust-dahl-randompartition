#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct CppParameters CppParameters;

typedef struct CrpParameters CrpParameters;

typedef struct EpaParameters EpaParameters;

typedef struct FixedPartitionParameters FixedPartitionParameters;

typedef struct FrpParameters FrpParameters;

typedef struct JlpParameters JlpParameters;

typedef struct LspParameters LspParameters;

typedef struct SpParameters SpParameters;

typedef struct UpParameters UpParameters;

typedef struct Rr_Sexp_Vector_Intsxp {
  const void *sexp_ptr;
  int32_t *data_ptr;
  int32_t len;
} Rr_Sexp_Vector_Intsxp;

typedef struct Rr_Sexp {
  const void *sexp_ptr;
} Rr_Sexp;

struct CppParameters *dahl_randompartition__cppparameters_new(int32_t n_items,
                                                              const int32_t *baseline_ptr,
                                                              double rate,
                                                              bool uniform,
                                                              double mass,
                                                              double discount,
                                                              bool use_vi,
                                                              double a);

void dahl_randompartition__cppparameters_free(struct CppParameters *obj);

struct CrpParameters *dahl_randompartition__crpparameters_new(int32_t n_items,
                                                              double mass,
                                                              double discount);

void dahl_randompartition__crpparameters_free(struct CrpParameters *obj);

struct EpaParameters *dahl_randompartition__epaparameters_new(int32_t n_items,
                                                              double *similarity_ptr,
                                                              const int32_t *permutation_ptr,
                                                              int32_t use_natural_permutation,
                                                              double mass,
                                                              double discount);

void dahl_randompartition__epaparameters_free(struct EpaParameters *obj);

struct FrpParameters *dahl_randompartition__frpparameters_new(int32_t n_items,
                                                              const int32_t *baseline_ptr,
                                                              const double *shrinkage_ptr,
                                                              const int32_t *permutation_ptr,
                                                              int32_t use_natural_permutation,
                                                              double mass,
                                                              double discount,
                                                              double power);

void dahl_randompartition__frpparameters_free(struct FrpParameters *obj);

struct LspParameters *dahl_randompartition__lspparameters_new(int32_t n_items,
                                                              const int32_t *baseline_ptr,
                                                              double rate,
                                                              const int32_t *permutation_ptr,
                                                              int32_t use_natural_permutation);

void dahl_randompartition__lspparameters_free(struct LspParameters *obj);

extern struct Rr_Sexp_Vector_Intsxp rrAllocVectorINTSXP(int32_t len);

extern double callRFunction_logIntegratedLikelihoodItem(const void *fn_ptr,
                                                        int32_t i,
                                                        struct Rr_Sexp_Vector_Intsxp indices,
                                                        const void *env_ptr);

extern double callRFunction_logIntegratedLikelihoodSubset(const void *fn_ptr,
                                                          struct Rr_Sexp_Vector_Intsxp indices,
                                                          const void *env_ptr);

extern double callRFunction_logLikelihoodItem(const void *fn_ptr,
                                              int32_t i,
                                              int32_t label,
                                              int32_t is_new,
                                              const void *env_ptr);

void dahl_randompartition__neal_algorithm3(int32_t n_updates_for_partition,
                                           int32_t n_items,
                                           int32_t *partition_ptr,
                                           int32_t prior_only,
                                           const void *log_posterior_predictive_function_ptr,
                                           const void *env_ptr,
                                           const int32_t *seed_ptr,
                                           int32_t prior_id,
                                           const void *prior_ptr);

void dahl_randompartition__neal_algorithm8(int32_t n_updates_for_partition,
                                           int32_t n_items,
                                           int32_t *partition_ptr,
                                           int32_t prior_only,
                                           const void *log_likelihood_function_ptr,
                                           const void *env_ptr,
                                           const int32_t *seed_ptr,
                                           int32_t prior_id,
                                           const void *prior_ptr,
                                           struct Rr_Sexp *map_ptr);

void dahl_randompartition__sample_partition(int32_t n_partitions,
                                            int32_t n_items,
                                            int32_t *partition_labels_ptr,
                                            const int32_t *seed_ptr,
                                            int32_t prior_id,
                                            const void *prior_ptr,
                                            bool randomize_permutation);

void dahl_randompartition__log_probability_of_partition(int32_t n_partitions,
                                                        int32_t n_items,
                                                        int32_t *partition_labels_ptr,
                                                        double *log_probabilities_ptr,
                                                        int32_t prior_id,
                                                        const void *prior_ptr);

struct FixedPartitionParameters *dahl_randompartition__fixedpartitionparameters_new(int32_t n_items,
                                                                                    const int32_t *clustering_ptr);

void dahl_randompartition__fixedpartitionparameters_free(struct FixedPartitionParameters *obj);

struct JlpParameters *dahl_randompartition__jlpparameters_new(int32_t n_items,
                                                              double mass,
                                                              const int32_t *permutation_ptr,
                                                              int32_t use_natural_permutation);

void dahl_randompartition__jlpparameters_free(struct JlpParameters *obj);

struct SpParameters *dahl_randompartition__trpparameters_new(int32_t n_items,
                                                             const int32_t *baseline_partition_ptr,
                                                             const double *shrinkage_ptr,
                                                             const int32_t *permutation_ptr,
                                                             int32_t use_natural_permutation,
                                                             int32_t baseline_distr_id,
                                                             const void *baseline_distr_ptr,
                                                             int32_t loss,
                                                             double a);

void dahl_randompartition__trpparameters_free(struct SpParameters *obj);

struct UpParameters *dahl_randompartition__upparameters_new(int32_t n_items);

void dahl_randompartition__upparameters_free(struct UpParameters *obj);
