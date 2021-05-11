#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <limits>
#include <mutex>
#include <queue>
#include <vector>

#include "vector_util.h"

using namespace tensorflow;

REGISTER_OP("Clustering")
    .Input("code: uint16")
    .Input("codebook: float32")
    .Input("n_cluster: int32")
    .Input("sample_size: int32")
    .Input("max_iter: int32")
    .Input("change_percentage_thr: float32")
    .Output("centroid: float32")
    .Output("assignment: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        return Status::OK();
    });

struct Codebook {
    const float *codebook_;
    int64 D_;
    int64 K_;
    int64 sub_dim_;

    Codebook(const float *codebook, int64 D, int64 K, int64 sub_dim)
        : codebook_(codebook), D_(D), K_(K), sub_dim_(sub_dim) {}

    inline const float GetCode(int64 D_index, int64 K_index, int m) const {
        return codebook_[D_index * (K_ * sub_dim_) + K_index * sub_dim_ + m];
    }

    inline const float* GetOffset(int64 D_index, int64 K_index, int m) const {
        return codebook_ + D_index * (K_ * sub_dim_) + K_index * sub_dim_ + m;
    }
};

template <typename T>
struct PQCode {
    const T *code_;
    int64 n_code_;
    int64 D_;

    PQCode(const T *code, int64 batch_size, int64 D)
        : code_(code), n_code_(batch_size), D_(D) {}

    inline const T GetCode(int64 index, int64 D_index) const {
        return code_[index * D_ + D_index];
    }

    inline const T* GetOffset(int64 index, int64 D_index) const {
        return code_ + index * D_ + D_index;
    }

};

struct Centroid {
    std::vector<float> centroid_;
    int   n_cluster_;
    int64 d_;
    int64 D_;
    int64 sub_dim_;

    Centroid(int64 n_cluster, int64 d, int64 D)
        : n_cluster_(n_cluster), d_(d), D_(D), sub_dim_(d/D) {
        centroid_.resize(n_cluster * d, 0.0);
    }

    void Reset() {
        std::fill(centroid_.begin(), centroid_.end(), 0.0);
    }

    inline float GetCode(int64 index, int64 D_index, int64 m) const {
        return centroid_[index * d_ + D_index * sub_dim_ + m];
    }

    inline float *GetOffset(int64 index, int64 D_index, int64 m) {
        return centroid_.data() + index * d_ + D_index * sub_dim_ + m;
    }

    inline const float *GetOffset(int64 index, int64 D_index, int64 m) const {
        return centroid_.data() + index * d_ + D_index * sub_dim_ + m;
    }

    inline void GetPQCode(const Codebook &codebook, uint16 *pq_code) const {
        for (int i = 0; i < n_cluster_; ++i) {
            for (int j = 0; j < D_; ++j) {
                float min_dist = std::numeric_limits<float>::max();
                uint16 code = 0;
                for (int k = 0; k < codebook.K_; ++k) {
                    float dist = fvec_L2sqr(GetOffset(i, j, 0), codebook.GetOffset(j, k, 0), sub_dim_);
                    if (dist < min_dist) {
                        min_dist = dist;
                        code = k;
                    }
                }
                pq_code[i * D_ + j] = code;
            }
        }
    }

    inline float ChangedRatio(const Centroid &rhs) const {
        float total_dist = 0.0f;
        float total_norm = 0.0f;
        for (int i = 0; i < n_cluster_; ++i) {
            total_dist += fvec_L2sqr(GetOffset(i, 0, 0), rhs.GetOffset(i, 0, 0), d_);
            total_norm += fvec_inner_product(GetOffset(i, 0, 0), GetOffset(i, 0, 0), d_);
        }
        return total_dist / total_norm / n_cluster_;
    }
};

class SampleGenerator {
 public:
    explicit SampleGenerator(int n_item) : n_item_(n_item) {
        sampling_index_.resize(n_item_);
        std::iota(sampling_index_.begin(), sampling_index_.end(), 0);
    }

    std::vector<int> GetSamples(size_t sample_size) {
        for (size_t i = 0; i + 1 < n_item_; i++) {
            int i2 = i + rand() % (n_item_ - i);
            std::swap(sampling_index_[i], sampling_index_[i2]);
        }

        sample_size = std::min(sample_size, (size_t)n_item_);
        std::vector<int> samples(sampling_index_.begin(), sampling_index_.begin() + sample_size);

        return samples;
    }

 private:
    std::vector<int> sampling_index_;
    int n_item_;
};

class ClusteringOp : public OpKernel {
 public:
    explicit ClusteringOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        std::chrono::duration<double> elapsed_1(0.0);
        std::chrono::duration<double> elapsed_2(0.0);

        const int64 batch_size = context->input(0).dim_size(0);
        const int64 D = context->input(0).dim_size(1);
        const int64 K = context->input(1).dim_size(1);
        const int64 sub_dim = context->input(1).dim_size(2);
        const int64 d = sub_dim * D;

        const uint16 *code_ptr = context->input(0).flat<uint16>().data();    // [batch_size, D]
        const float *codebook_ptr = context->input(1).flat<float>().data();  // [D, K, sub_dim]

        const int n_cluster = context->input(2).scalar<int>()(0);
        int sample_size = context->input(3).scalar<int>()(0);
        const int max_iter = context->input(4).scalar<int>()(0);
        const float change_percentage_thr = context->input(5).scalar<float>()(0);

        LOG(INFO) << "batch_size:" << batch_size << ", D:" << D << ", K:" << K << ", sub_dim:" << sub_dim << ", d:" << d;
        LOG(INFO) << "n_cluster:" << n_cluster << ", max_iter:" << max_iter;
        CHECK(n_cluster > 0);
        CHECK(max_iter > 0);
        CHECK(change_percentage_thr > 0);

        PQCode<uint16> code(code_ptr, batch_size, D);
        Codebook codebook(codebook_ptr, D, K, sub_dim);
        Centroid centroid(n_cluster, d, D);

        // Create an output tensor
        Tensor *centroid_out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{n_cluster, d},
                                                         &centroid_out));
        auto centroid_flat = centroid_out->flat<float>();

        Tensor *assignment = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{batch_size},
                                                         &assignment));
        auto assignment_flat = assignment->flat<int64>();

        // Precompute L2 square between each subcode.
        // Later we compute the PQ codes of centroids, and approximate the L2 square quickly.
        std::vector<uint16> centroid_pq_code_vec(n_cluster * D);
        PQCode<uint16> centroid_pq_code(centroid_pq_code_vec.data(), n_cluster, D);
        std::vector<float> L2sqr_table(D * K * K);
        for (size_t i = 0; i < D; ++i) {
            for (size_t j = 0; j < K; ++j) {
                for (size_t k = 0; k < K; ++k) {
                    float l2sqr = fvec_L2sqr(codebook.GetOffset(i, j, 0), codebook.GetOffset(i, k, 0), sub_dim);
                    L2sqr_table[i * (K * K) + j * K + k] = l2sqr;
                }
            }
        }

        // allocate and initialize two variables: centroid and assignment
        std::vector<int> assignments;
        assignments.resize(batch_size);
        for (int64 i = 0; i < batch_size; ++i) {
            assignments[i] = i % n_cluster;
        }
        std::random_shuffle(assignments.begin(), assignments.end());
        SampleGenerator sample_generator((int)batch_size);
        auto samples = sample_generator.GetSamples(sample_size);
        ComputeCentroid(samples, assignments, codebook, code, &centroid);

        // while assignment is still changing or reaching max iterations
        for (int iter = 0; iter < max_iter; ++iter) {
            auto local_start = std::chrono::high_resolution_clock::now();
            auto samples = sample_generator.GetSamples(sample_size);

            // step 1: compute assignments
            float avg_distance;
            int changed = ComputeAssignment(samples, codebook, code, centroid, L2sqr_table, iter, &assignments, &avg_distance);
            auto local_end = std::chrono::high_resolution_clock::now();
            elapsed_1 += (local_end - local_start);

            LOG(INFO) << "Elapsed time, compute assignment: " << elapsed_1.count() * 1000 << " ms";
            float change_percentage = (float)changed / (float)sample_size;
            LOG(INFO) << "iter " << iter << " : average distance = " << avg_distance
                      << ", assignment changed = " << changed << " (" << change_percentage << ")";

            // step 2: compute centroid
            local_start = std::chrono::high_resolution_clock::now();
            Centroid old_centroid = centroid;
            ComputeCentroid(samples, assignments, codebook, code, &centroid);

            local_end = std::chrono::high_resolution_clock::now();
            elapsed_2 += (local_end - local_start);

            LOG(INFO) << "Elapsed time, compute centroid: " << elapsed_2.count() * 1000 << " ms";

            float centroid_change_ratio = old_centroid.ChangedRatio(centroid);
            LOG(INFO) << "iter " << iter << " : centroid_change_ratio = " << centroid_change_ratio << "\n";
            if (centroid_change_ratio < change_percentage_thr) {
                if (sample_size < batch_size) {
                    sample_size = batch_size;
                } else {
                    // probably converged but just changed by numerical instability.
                    break;
                }
            }
        }

        // copy centroid vector
        std::copy_n(centroid.centroid_.begin(), n_cluster * d, centroid_flat.data());

        // copy assignment vector
        for (int i = 0; i < batch_size; ++i) {
            assignment_flat(i) = assignments[i];
        }

        LOG(INFO) << "Elapsed time 1: " << elapsed_1.count() * 1000 << " ms\n";
        LOG(INFO) << "Elapsed time 2: " << elapsed_2.count() * 1000 << " ms\n";
    }


    void ComputeCentroid(const std::vector<int> &samples, const std::vector<int> &assignments, const Codebook& codebook,
                        const PQCode<uint16> &code, Centroid *centroid) {
        int64 n_cluster = centroid->n_cluster_;
        int64 D = centroid->D_;
        int64 sub_dim = centroid->sub_dim_;
        int64 d = centroid->d_;
        std::vector<int> counter(n_cluster, 0);
        centroid->Reset();

#pragma omp parallel for num_threads(32)
        for (int si = 0; si < samples.size(); ++si) {
            int i = samples[si];
            auto assign = assignments[i];
            ++counter[assign];
            for (int j = 0; j < D; ++j) {
                for (int m = 0; m < sub_dim; ++m) {
                    *(centroid->GetOffset(assign, j, m)) += codebook.GetCode(j, code.GetCode(i, j), m);
                }
            }
        }

        for (int c = 0; c < n_cluster; ++c) {
            auto start = centroid->GetOffset(c, 0, 0);
            if (counter[c] > 0) {
                for (int j = 0; j < d; ++j) {
                    *(start + j) /= (float)counter[c];
                }
            }
        }

    }

    int ComputeAssignment(const std::vector<int> &samples, const Codebook& codebook, const PQCode<uint16> &code,
                          const Centroid &centroid, const std::vector<float> &L2sqr_table, int iter, std::vector<int> *assignments,
                          float *avg_distance) {

        int n_cluster = centroid.n_cluster_;
        int64 D = centroid.D_;
        int64 sub_dim = centroid.sub_dim_;
        int64 d = centroid.d_;
        int64 K = codebook.K_;

        std::vector<uint16> centroid_pq_code_vec(n_cluster * D);
        PQCode<uint16> centroid_pq_code(centroid_pq_code_vec.data(), n_cluster, D);

        centroid.GetPQCode(codebook, centroid_pq_code_vec.data());
        int n_probe = std::min(n_cluster, 8);  // TODO: hongwei.shen1, maybe calculate n_probe from other paramters,
                                               //       or get it from input parameters

        int search_centroid_num = std::max(n_cluster / (iter*iter+1), n_probe);
        search_centroid_num = std::min(search_centroid_num, n_cluster);
        int changed = 0;
        float total_dist = 0.0f;
#pragma omp parallel for num_threads(32) reduction(+: changed, total_dist)
        for (int si = 0; si < samples.size(); ++si) {
            int i = samples[si];
            // Compute appproximate L2 squre between current point and all centroid
            // using precomputed L2 square between each subcode.
            std::vector<std::pair<float, int>> dist_vec;
            dist_vec.reserve(n_cluster);
            for (int c = 0; c < n_cluster; ++c) {
                float dist = 0.0f;
                for (int j = 0; j < D; ++j) {
                    dist += L2sqr_table[j * K * K + code.GetCode(i, j) * K + centroid_pq_code.GetCode(c, j)];
                }
                dist_vec.emplace_back(dist, c);
            }

            // Find the nearest centroid from the top centroids in dist_vec
            // search_centroid_num is large at begining iteration,
            //   then it is smaller when the clusering is more refined at later iterations
            std::nth_element(dist_vec.begin(), dist_vec.begin() + search_centroid_num, dist_vec.end());
            std::sort(dist_vec.begin(), dist_vec.begin() + search_centroid_num);  // sort in ascending order of distance

            float min_dist = std::numeric_limits<float>::max();
            int min_centroid = -1;
            float item[d];  // buffer to store item vector
            size_t mem_copy_size = sizeof(float) * sub_dim;
            for (int m = 0; m < search_centroid_num; ++m) {
                int c = dist_vec[m].second;
                // Calcuate actual L2 squre to current centroid,
                for (int j = 0; j < D; ++j) {
                    const float *p = codebook.GetOffset(j, code.GetCode(i, j), 0);
                    memcpy(item + j * sub_dim, p, mem_copy_size);
                }
                float dist = fvec_L2sqr(item, centroid.GetOffset(c, 0, 0), d);

                if (dist + 1e-7 < min_dist) {
                    min_dist = dist;
                    min_centroid = c;
                }
            }

            if (min_centroid != (*assignments)[i]) {
                ++changed;
                (*assignments)[i] = min_centroid;
            }
            total_dist = total_dist + sqrt(min_dist);
        }

        *avg_distance = total_dist / samples.size();
        return changed;
    }
};

REGISTER_KERNEL_BUILDER(Name("Clustering").Device(DEVICE_CPU), ClusteringOp);
