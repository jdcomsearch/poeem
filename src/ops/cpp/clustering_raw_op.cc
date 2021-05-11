#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
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

REGISTER_OP("ClusteringRaw")
    .Input("data: float32")
    .Input("n_cluster: int32")
    .Input("max_iter: int32")
    .Input("change_percentage_thr: float32")
    .Input("verbose: int32" )
    .Output("centroid: float32")
    .Output("assignment: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        return Status::OK();
    });

struct Data {
    const float *data_ptr_;
    int batch_size_;
    int64 d_;

    Data(const float *data_ptr, int batch_size, int64 d)
        : data_ptr_(data_ptr), batch_size_(batch_size), d_(d) {}

    inline const float *GetOffset(int64 index, int64 d_index) const {
        return data_ptr_ + index * d_ + d_index;
    }
};

struct Centroid {
    std::vector<float> centroid_;
    int n_cluster_;
    int64 d_;

    Centroid(int64 n_cluster, int64 d)
        : n_cluster_(n_cluster), d_(d) {
        centroid_.resize(n_cluster * d, 0.0);
    }

    void Reset() {
        std::fill(centroid_.begin(), centroid_.end(), 0.0);
    }

    inline float *GetOffset(int64 index, int64 d_index) {
        return centroid_.data() + index * d_ + d_index;
    }

    inline const float *GetOffset(int64 index, int64 d_index) const {
        return centroid_.data() + index * d_ + d_index;
    }
};

class ClusteringRawOp : public OpKernel
{
public:
    explicit ClusteringRawOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const int64 batch_size = context->input(0).dim_size(0);
        const int64 d = context->input(0).dim_size(1);

        const float *data_ptr = context->input(0).flat<float>().data(); // [batch_size, d]

        const int n_cluster = context->input(1).scalar<int>()(0);
        const int max_iter = context->input(2).scalar<int>()(0);
        const float change_percentage_thr = context->input(3).scalar<float>()(0);
        const int verbose = context->input(4).scalar<int>()(0);

        if (verbose >= 2) {
            LOG(INFO) << "batch_size:" << batch_size << ", d:" << d << ", n_cluster:" << n_cluster << ", max_iter:" << max_iter;
        }

        CHECK(n_cluster > 0);
        CHECK(max_iter > 0);
        CHECK(change_percentage_thr > 0);

        Data data(data_ptr, batch_size, d);
        Centroid centroid(n_cluster, d);

        // Create an output tensor
        Tensor *centroid_out = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{n_cluster, d},
                                                         &centroid_out));
        auto centroid_flat = centroid_out->flat<float>();

        Tensor *assignment = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{batch_size},
                                                         &assignment));
        auto assignment_flat = assignment->flat<int64>();

        // allocate and initialize two variables: centroid and assignment
        std::vector<int> assignments;
        assignments.resize(batch_size);
        for (int64 i = 0; i < batch_size; ++i) {
            assignments[i] = i % n_cluster;
        }
        std::random_shuffle(assignments.begin(), assignments.end());
        ComputeCentroid(assignments, data, &centroid);

        // while assignment is still changing or reaching max iterations
        for (int iter = 0; iter < max_iter; ++iter) {
            // step 1: compute assignments
            float avg_distance;
            int changed = ComputeAssignment(data, centroid, iter, &assignments, &avg_distance);

            float change_percentage = (float)changed / (float)batch_size;
            if (verbose >= 3) {
                LOG(INFO) << "iter " << iter << " : average distance = " << avg_distance
                          << ", assignment changed = " << changed << " (" << change_percentage << ")";
            }
            if (change_percentage < change_percentage_thr) {
                if (verbose >= 1) {
                    LOG(INFO) << "Converged at iter " << iter << " : average distance = " << avg_distance
                              << ", assignment changed = " << changed << " (" << change_percentage << ")"
                              << ", assignment frequency histogram = " << FrequencyHistogram(assignments);
                }
                break;
            }

            // step 2: compute centroid
            Centroid old_centroid = centroid;
            ComputeCentroid(assignments, data, &centroid);
        }

        // copy centroid vector
        std::copy_n(centroid.centroid_.begin(), n_cluster * d, centroid_flat.data());

        // copy assignment vector
        for (int i = 0; i < batch_size; ++i) {
            assignment_flat(i) = assignments[i];
        }
    }

    void ComputeCentroid(const std::vector<int> &assignments, const Data &data, Centroid *centroid) {
        int64 n_cluster = centroid->n_cluster_;
        int64 d = centroid->d_;
        int64 batch_size = data.batch_size_;
        std::vector<int> counter(n_cluster, 0);
        centroid->Reset();

#pragma omp parallel for num_threads(32)
        for (int i = 0; i < batch_size; ++i) {
            auto assign = assignments[i];
            ++counter[assign];
            for (int j = 0; j < d; ++j) {
                *(centroid->GetOffset(assign, j)) += *(data.GetOffset(i, j));
            }
        }

        for (int c = 0; c < n_cluster; ++c) {
            auto start = centroid->GetOffset(c, 0);
            if (counter[c] > 0) {
                for (int j = 0; j < d; ++j) {
                    *(start + j) /= (float)counter[c];
                }
            }
        }
    }

    std::string FrequencyHistogram(const std::vector<int>& assignments) {
        std::map<int, int> counter;
        for (const int a : assignments) {
            ++counter[a];
        }
        std::map<int, int> histogram;
        for (const auto& it: counter) {
            ++histogram[it.second];
        }
        std::string ret;
        ret.reserve(assignments.size() * 10);
        int cnt = 0;
        for (int i = 0; i < assignments.size(); ++i) {
            auto it = histogram.find(i);
            if (it == histogram.end()) continue;
            absl::StrAppend(&ret, "(", it->first, ", ", it->second, ") ");
            if (++cnt >= histogram.size()) break;
        }
        return ret;
    }

    int ComputeAssignment(const Data &data,
                          const Centroid &centroid,
                          int iter,
                          std::vector<int> *assignments,
                          float *avg_distance) {

        int n_cluster = centroid.n_cluster_;
        int64 d = centroid.d_;
        int64 batch_size = data.batch_size_;

        int changed = 0;
        float total_dist = 0.0f;
#pragma omp parallel for num_threads(32) reduction(+ \
                                                   : changed, total_dist)
        for (int i = 0; i < batch_size; ++i) {
            // Compute appproximate L2 squre between current point and all centroid
            // using precomputed L2 square between each subcode.
            int64 min_centroid = -1;
            float min_dist = std::numeric_limits<float>::max();
            for (int c = 0; c < n_cluster; ++c) {
                float dist = fvec_L2sqr(data.GetOffset(i, 0), centroid.GetOffset(c, 0), d);
                if (dist < min_dist) {
                    min_centroid = c;
                    min_dist = dist;
                }
            }

            if (min_centroid != (*assignments)[i]) {
                ++changed;
                (*assignments)[i] = min_centroid;
            }
            total_dist = total_dist + sqrt(min_dist);
        }

        *avg_distance = total_dist / batch_size;
        return changed;
    }
};

REGISTER_KERNEL_BUILDER(Name("ClusteringRaw").Device(DEVICE_CPU), ClusteringRawOp);
