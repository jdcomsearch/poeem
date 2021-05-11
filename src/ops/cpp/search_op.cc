#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "vector_util.h"

using namespace tensorflow;

static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */

struct ItemDistance {
    float dist;
    uint32 item_id;
    bool operator<(const ItemDistance &rhs) const { return dist < rhs.dist; }
    bool operator>(const ItemDistance &rhs) const { return dist > rhs.dist; }
};

enum MetricType { L2Norm = 0, InnerProduct = 1, Cosine = 2 };

class Centroids {
   public:
    Centroids(int dim, const std::vector<float> *centroid_vec)
        : d_(dim), centroid_vec_(centroid_vec) {
        n_centroid_ = centroid_vec_->size() / d_;
    }

    void GetNearestN(const float *query, int n_probe,
                     std::vector<int> *center_ids, int num_thread = 1) const {
        const float *centers = centroid_vec_->data();
        std::vector<std::pair<float, int>> centroid_dist(n_centroid_);
#pragma omp parallel for if (num_thread > 1) num_threads(num_thread)
        for (int i = 0; i < n_centroid_; ++i) {
            float dist = fvec_L2sqr(query, centers + i * d_, d_);
            centroid_dist[i] = std::make_pair(dist, i);
        }
        std::nth_element(centroid_dist.begin(), centroid_dist.begin() + n_probe,
                         centroid_dist.end());
        std::sort(centroid_dist.begin(), centroid_dist.begin() + n_probe);
        center_ids->resize(n_probe);
        for (int i = 0; i < n_probe; ++i) {
            (*center_ids)[i] = centroid_dist[i].second;
        }
    }

    void GetResidual(const float *query, int centroid_id,
                     float *residual) const {
        const float *center = centroid_vec_->data() + centroid_id * d_;
        fvec_substraction(query, center, d_, residual);
    }

    inline const float *GetOffset(int centroid_id) const {
        return centroid_vec_->data() + centroid_id * d_;
    }

   private:
    const std::vector<float> *centroid_vec_;
    int n_centroid_;
    int d_;
};

// Class of Asymmetric Distance Computation
class ADC {
   public:
    ADC(int d, int K, int D, bool use_residual, MetricType metric_type,
        const std::vector<uint8> *code8, const std::vector<uint16> *code16,
        const std::vector<float> *codebook, const std::vector<float> *norm)
        : d_(d),
          K_(K),
          D_(D),
          sub_dim_(d / D),
          use_residual_(use_residual),
          metric_type_(metric_type),
          code8_(code8),
          code16_(code16),
          codebook_(codebook),
          norm_(norm) {}

    // compute lookup table (query embedding to codebook distance)
    inline void PreCompute(const float *query) {
        query_norm_ = fvec_inner_product(query, query, d_);
        query_item_inner_product_.resize(D_ * K_);
        for (int i = 0; i < D_; ++i) {
            const float *q = query + i * sub_dim_;
            const float *c = codebook_->data() + i * sub_dim_ * K_;
            for (int j = 0; j < K_; ++j) {
                query_item_inner_product_[i * K_ + j] =
                    fvec_inner_product(q, c + j * sub_dim_, sub_dim_);
            }
        }
    }

    inline void PreCompute(const float *query, const Centroids &centroids,
                           int centroid_id) {
        query_norm_ = fvec_inner_product(query, query, d_);
        query_item_inner_product_.resize(D_ * K_);
        for (int i = 0; i < D_; ++i) {
            const float *q = query + i * sub_dim_;
            const float *c = codebook_->data() + i * sub_dim_ * K_;
            const float *center =
                centroids.GetOffset(centroid_id) + i * sub_dim_;
            float original[sub_dim_];
            for (int j = 0; j < K_; ++j) {
                fvec_addition(c + j * sub_dim_, center, sub_dim_, original);
                query_item_inner_product_[i * K_ + j] =
                    fvec_inner_product(q, original, sub_dim_);
            }
        }
    }

    void BatchMetric(const std::vector<uint32> &item_ids,
                     ItemDistance *p_item_dist) {
        if (metric_type_ == L2Norm) {
            BatchL2Norm(item_ids, p_item_dist);
        } else if (metric_type_ == InnerProduct) {
            BatchInnerProduct(item_ids, p_item_dist);
        } else {  // Cosine
            BatchCosine(item_ids, p_item_dist);
        }
    }

    void BatchL2Norm(const std::vector<uint32> &item_ids,
                     ItemDistance *p_item_dist) {
        if (K_ <= 256) {
            for (const auto item_id : item_ids) {
                p_item_dist->dist = query_norm_ + (*norm_)[item_id] -
                                    2 * InnerProduct8(item_id);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        } else {
            for (const auto item_id : item_ids) {
                p_item_dist->dist = query_norm_ + (*norm_)[item_id] -
                                    2 * InnerProduct16(item_id);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        }
    }

    void BatchInnerProduct(const std::vector<uint32> &item_ids,
                           ItemDistance *p_item_dist) {
        if (K_ <= 256) {
            for (const auto item_id : item_ids) {
                p_item_dist->dist = InnerProduct8(item_id);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        } else {
            for (const auto item_id : item_ids) {
                p_item_dist->dist = InnerProduct16(item_id);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        }
    }

    void BatchCosine(const std::vector<uint32> &item_ids,
                     ItemDistance *p_item_dist) {
        if (K_ <= 256) {
            for (const auto item_id : item_ids) {
                p_item_dist->dist =
                    InnerProduct8(item_id) / (query_norm_ * (*norm_)[item_id]);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        } else {
            for (const auto item_id : item_ids) {
                p_item_dist->dist =
                    InnerProduct16(item_id) / (query_norm_ * (*norm_)[item_id]);
                p_item_dist->item_id = item_id;
                ++p_item_dist;
            }
        }
    }

   private:
    inline float InnerProduct8(int item_id) const {
        float ip = 0.0f;
        const auto *p = code8_->data() + item_id * D_;
        for (int i = 0; i < D_; ++i) {
            ip += query_item_inner_product_[i * K_ + p[i]];
        }
        return ip;
    }

    inline float InnerProduct16(int item_id) const {
        float ip = 0.0f;
        const auto *p = code16_->data() + item_id * D_;
        for (int i = 0; i < D_; ++i) {
            ip += query_item_inner_product_[i * K_ + p[i]];
        }
        return ip;
    }

    const int d_;
    const int K_;
    const int D_;
    const int sub_dim_;
    const bool use_residual_;
    const MetricType metric_type_;
    const std::vector<uint8> *code8_;
    const std::vector<uint16> *code16_;
    const std::vector<float> *codebook_;
    const std::vector<float> *norm_;
    std::vector<float> query_item_inner_product_;
    float query_norm_;
};

class Index : public ResourceBase {
   public:
    Index(OpKernelContext *ctx, OpKernel *kernel) {}

    /*************
     *  Parsing the following format of index file in binary format
     *
     *  use_residual               (int8)
     *  n_batch d n_cluster K D    (int64)
     *  codebook                   (float32)
     *  item_id                    (int64)
     *  item_norm                  (float32)
     *  item_code                  (uint8 or uint16)
     *  centroid                   (float32)
     *  assignment.indices[:, 1]   (int64)
     *
     *************/
    Status InitializeFromFile(OpKernelContext *context,
                              const std::string &index_file_name) {
        std::unique_ptr<RandomAccessFile> index_file;
        context->env()->NewRandomAccessFile(index_file_name, &index_file);
        std::unique_ptr<io::InputBuffer> input_buffer;
        input_buffer.reset(
            new io::InputBuffer(index_file.get(), kInputBufferSize));

        size_t bytes_read;
        int8_t tmp8;
        TF_RETURN_IF_ERROR(input_buffer->ReadNBytes(
            sizeof(int8_t), (char *)&tmp8, &bytes_read));
        use_residual_ = bool(tmp8);
        int64 tmp;
        TF_RETURN_IF_ERROR(
            input_buffer->ReadNBytes(sizeof(int64), (char *)&tmp, &bytes_read));
        n_item_ = tmp;
        TF_RETURN_IF_ERROR(
            input_buffer->ReadNBytes(sizeof(int64), (char *)&d_, &bytes_read));
        TF_RETURN_IF_ERROR(input_buffer->ReadNBytes(
            sizeof(int64), (char *)&n_cluster_, &bytes_read));
        TF_RETURN_IF_ERROR(
            input_buffer->ReadNBytes(sizeof(int64), (char *)&K_, &bytes_read));
        TF_RETURN_IF_ERROR(
            input_buffer->ReadNBytes(sizeof(int64), (char *)&D_, &bytes_read));
        LOG(INFO) << "use_residual = " << use_residual_
                  << "; n_item = " << n_item_ << "; d = " << d_
                  << "; n_cluster = " << n_cluster_ << "; K = " << K_
                  << "; D = " << D_;

        ReadTensor(D_, K_ * d_ / D_, input_buffer, &codebook_);
        ReadTensor(n_item_, 1, input_buffer, &external_id_);
        ReadTensor(n_item_, 1, input_buffer, &norm_);
        if (K_ <= 256) {
            ReadTensor(n_item_, D_, input_buffer, &code8_);
        } else {
            ReadTensor(n_item_, D_, input_buffer, &code16_);
        }
        ReadTensor(n_cluster_, d_, input_buffer, &centroid_vec_);
        ReadTensor(n_item_, 1, input_buffer, &assignment_);

        // prepare cluster member lookup table from assignment sparse tensor
        cluster_item_ids_.resize(n_cluster_);
        for (uint32 i = 0; i < n_item_; ++i) {
            if (norm_[i] != 0.0f) {
                uint32 cluster_id = assignment_[i];
                cluster_item_ids_[cluster_id].push_back(i);
            }
        }

        for (int i = 0; i < cluster_item_ids_.size(); ++i) {
            LOG(INFO) << "cluster " << i << ":" << cluster_item_ids_[i].size();
        }

        return Status::OK();
    }

    void NNSearch(OpKernelContext *context, const Tensor &query_tensor,
                  int n_neighbor, int n_probe, MetricType metric_type) {
        std::chrono::duration<double> elapsed_1(0.0);
        std::chrono::duration<double> elapsed_2(0.0);
        std::chrono::duration<double> elapsed_3(0.0);
        auto start = std::chrono::high_resolution_clock::now();

        const int64 batch_size = query_tensor.dim_size(0);
        const int64 num_head = query_tensor.dim_size(1);
        const int64 d = query_tensor.dim_size(2);
        LOG(INFO) << ">>> query, batch_size:" << batch_size
                  << ", num_head:" << num_head << ", d:" << d;

        // Create an output tensor
        Tensor *output_neighbors = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape{batch_size, n_neighbor},
                                    &output_neighbors));
        auto neighbors = output_neighbors->flat<int64>();
        Tensor *output_scores = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    1, TensorShape{batch_size, n_neighbor},
                                    &output_scores));
        auto scores = output_scores->flat<float>();

        int n_centroids = centroid_vec_.size() / d;
        int sub_dim = d_ / D_;
        CHECK(d == d_);
        CHECK(d == D_ * sub_dim);
        const float *codebook = codebook_.data();

        const float *querys =
            query_tensor.flat<float>().data();  // d-dimension array
        int num_thread = round((n_item_ / n_cluster_ * n_probe * d) / 2.5e6);
        num_thread = std::min(n_probe / 2, num_thread);
        LOG(INFO) << "step 4";
        if (verbose_) {
            LOG(INFO) << "use_residual_:" << use_residual_
                      << ", n_item_:" << n_item_
                      << ", n_neighbor:" << n_neighbor
                      << ", n_probe:" << n_probe
                      << ", batch_size:" << batch_size
                      << ", metric_type:" << metric_type;
            LOG(INFO) << "d:" << d << ", sub_dim:" << sub_dim << ", d_:" << d_
                      << ", D:" << D_ << ", K_:" << K_
                      << ", n_cluser:" << n_cluster_;
            LOG(INFO) << "code8.size():" << code8_.size()
                      << ", code16.size():" << code16_.size()
                      << ", codebook_.size:" << codebook_.size()
                      << ", centroid_vec_.size:" << centroid_vec_.size()
                      << ", assignment_.size:" << assignment_.size();
            LOG(INFO) << "num_thread for finding the nearest n_probe centroids:"
                      << num_thread;
            LOG(INFO) << "num_thread for searching each vector in n_probe "
                         "centroids:"
                      << num_thread * 2;
        }

        std::vector<ItemDistance> distances;
        distances.reserve(
            81920);  // TODO: calculate a number from the input parameters
        for (int bi = 0; bi < batch_size; ++bi) {
            int total_item_num = 0;
            int query_idx = bi * num_head;
            for (int head = 0; head < num_head; ++head) {
                query_idx += head;
                const float *query = querys + query_idx * d_;
                if (fvec_inner_product(query, query, d_) == 0.0f) {
                    for (int idx = query_idx * n_neighbor;
                         idx < (query_idx + 1) * n_neighbor; ++idx) {
                        neighbors(idx) = -1;
                    }
                    continue;
                }

                // step 1: find the nearest n_probe centroids
                auto local_start = std::chrono::high_resolution_clock::now();
                Centroids centroids(d_, &centroid_vec_);
                std::vector<int> center_ids(n_probe);
                centroids.GetNearestN(query, n_probe, &center_ids, num_thread);
                auto local_end = std::chrono::high_resolution_clock::now();
                elapsed_1 += (local_end - local_start);

                // step 2: exhaustive search within n_probe clusters
                local_start = std::chrono::high_resolution_clock::now();

                ADC adc(d_, K_, D_, use_residual_, metric_type, &code8_,
                        &code16_, &codebook_, &norm_);
                if (!use_residual_) {
                    adc.PreCompute(query);
                }

                std::vector<int> cluser_idx_start(center_ids.size());
                int idx_start = 0;
                for (int ci = 0; ci < center_ids.size(); ++ci) {
                    cluser_idx_start[ci] = idx_start;
                    idx_start += cluster_item_ids_[center_ids[ci]].size();
                }
                total_item_num += idx_start;
                distances.resize(total_item_num);

#pragma omp parallel for if (num_thread > 1) num_threads(num_thread * 2)
                for (int ci = 0; ci < center_ids.size(); ++ci) {
                    const auto centroid_id = center_ids[ci];
                    const auto &item_ids = cluster_item_ids_[centroid_id];
                    auto *p_item_dist = distances.data() +
                                        (total_item_num - idx_start) +
                                        cluser_idx_start[ci];

                    if (!use_residual_) {
                        adc.BatchMetric(item_ids, p_item_dist);
                    } else {
                        ADC local_adc = adc;
                        local_adc.PreCompute(query, centroids, centroid_id);
                        local_adc.BatchMetric(item_ids, p_item_dist);
                    }
                }
                local_end = std::chrono::high_resolution_clock::now();
                elapsed_2 += (local_end - local_start);
            }
            // Step 3: output results
            auto local_start = std::chrono::high_resolution_clock::now();
            int actual_n_neighbor = std::min(n_neighbor, total_item_num);
            if (metric_type == L2Norm) {
                std::nth_element(distances.begin(),
                                 distances.begin() + actual_n_neighbor,
                                 distances.begin() + total_item_num);
                std::sort(distances.begin(),
                          distances.begin() + actual_n_neighbor);
            } else {
                std::nth_element(distances.begin(),
                                 distances.begin() + actual_n_neighbor,
                                 distances.begin() + total_item_num,
                                 std::greater<ItemDistance>());
                std::sort(distances.begin(),
                          distances.begin() + actual_n_neighbor,
                          std::greater<ItemDistance>());
            }
            int idx = query_idx / num_head * n_neighbor;
            for (auto it = distances.begin();
                 it != distances.begin() + actual_n_neighbor; ++it) {
                neighbors(idx) =
                    external_id_[it->item_id];  // return external item id
                scores(idx) = it->dist;
                ++idx;
            }
            for (; idx < (query_idx / num_head + 1) * n_neighbor; ++idx) {
                neighbors(idx) = -1;
            }

            auto local_end = std::chrono::high_resolution_clock::now();
            elapsed_3 += (local_end - local_start);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        if (verbose_) {
            LOG(INFO) << "Elapsed time: " << elapsed.count() * 1000 << " ms";
            LOG(INFO) << "Elapsed time 1: " << elapsed_1.count() * 1000
                      << " ms";
            LOG(INFO) << "Elapsed time 2: " << elapsed_2.count() * 1000
                      << " ms";
            LOG(INFO) << "Elapsed time 3: " << elapsed_3.count() * 1000
                      << " ms";
        }
    }

    Index* GetInitializableIndex() {
        return this;
    }

    string DebugString() const {
        return "";
    }

   private:
    template <typename T>
    Status ReadTensor(size_t n_rows, size_t row_size,
                      std::unique_ptr<io::InputBuffer> &input_buffer,
                      std::vector<T> *out) {
        size_t row_char_size = sizeof(T) * row_size;
        out->resize(n_rows * row_size);
        size_t bytes_read;
        char *dst = (char *)(out->data());
        for (size_t i = 0; i < n_rows; ++i, dst += row_char_size) {
            TF_RETURN_IF_ERROR(
                input_buffer->ReadNBytes(row_char_size, dst, &bytes_read));
        }
        return Status::OK();
    }

    bool use_residual_ = true;
    uint32 n_item_;
    int64 d_;
    int64 n_cluster_;
    int64 K_;
    int64 D_;
    std::vector<int64> external_id_;
    std::vector<float> norm_;
    std::vector<uint8> code8_;
    std::vector<uint16> code16_;
    std::vector<float> codebook_;
    std::vector<float> centroid_vec_;
    std::vector<uint32> assignment_;
    std::vector<std::vector<uint32>> cluster_item_ids_;
    bool verbose_ = true;
};

REGISTER_OP("Index")
    .Output("index_handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("use_node_name_sharing: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        // c->set_output(0, {2});
        return Status::OK();
    });

class IndexOp : public OpKernel {
   public:
    // ctx is not owned by this class.
    explicit IndexOp(OpKernelConstruction *ctx)
        : OpKernel(ctx), index_handle_set_(false) {
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_persistent(tensorflow::DT_STRING,
                                                tensorflow::TensorShape({2}),
                                                &index_handle_, nullptr));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_node_name_sharing",
                                         &use_node_name_sharing_));
    }

    // ctx is not owned by this function.
    void Compute(OpKernelContext *ctx) override {
        mutex_lock l(mu_);

        if (!index_handle_set_) {
            
            OP_REQUIRES_OK(
                ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                 use_node_name_sharing_));
        }

        auto creator = [ctx, this](Index **ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            Index *index = new Index(ctx, this);
            if (!ctx->status().ok()) {
                index->Unref();
                return ctx->status();
            }
            if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    index->MemoryUsed() + index_handle_.AllocatedBytes());
            }
            *ret = index;
            return Status::OK();
        };

        Index *index = nullptr;
        OP_REQUIRES_OK(
            ctx, cinfo_.resource_manager()->template LookupOrCreate<Index>(
                     cinfo_.container(), cinfo_.name(), &index, creator));
        core::ScopedUnref unref_me(index);

        if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
            Tensor *handle;
            OP_REQUIRES_OK(ctx,
                           ctx->allocate_output(0, TensorShape({}), &handle));
            handle->scalar<ResourceHandle>()() =
                MakeResourceHandle<Index>(
                    ctx, cinfo_.container(), cinfo_.name());
        } else {
            if (!index_handle_set_) {
                auto h =
                    index_handle_.AccessTensor(ctx)->template flat<tstring>();
                h(0) = cinfo_.container();
                h(1) = cinfo_.name();
            }
            ctx->set_output_ref(0, &mu_, index_handle_.AccessTensor(ctx));
        }
        index_handle_set_ = true;
    }

    ~IndexOp() override {
        // If the index object was not shared, delete it.
        if (index_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
            if (!cinfo_.resource_manager()
                     ->template Delete<Index>(cinfo_.container(), cinfo_.name())
                     .ok()) {
                // Do nothing; the resource can have been deleted by session
                // resets.
            }
        }
    }

   private:
    mutex mu_;
    PersistentTensor index_handle_ GUARDED_BY(mu_);
    bool index_handle_set_ GUARDED_BY(mu_);
    ContainerInfo cinfo_;
    bool use_node_name_sharing_;

    TF_DISALLOW_COPY_AND_ASSIGN(IndexOp);
};

Status GetTableHandle(const string &input_name, OpKernelContext *ctx,
                      string *container, string *index_handle) {
    {
        mutex *mu;
        TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
        mutex_lock l(*mu);
        Tensor tensor;
        TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
        if (tensor.NumElements() != 2) {
            return errors::InvalidArgument(
                "Lookup table handle must be scalar, but had shape: ",
                tensor.shape().DebugString());
        }
        auto h = tensor.flat<tstring>();
        *container = h(0);
        *index_handle = h(1);
    }
    return Status::OK();
}

Status GetIndex(const string &input_name, OpKernelContext *ctx, Index **index) {
    string container;
    string index_handle;
    DataType handle_dtype;
    TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
    if (handle_dtype == DT_RESOURCE) {
        ResourceHandle handle;
        TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
        return LookupResource(ctx, handle, index);
    } else {
        TF_RETURN_IF_ERROR(
            GetTableHandle(input_name, ctx, &container, &index_handle));
        return ctx->resource_manager()->Lookup(container, index_handle, index);
    }
}

REGISTER_OP("InitializeIndexFromFile")
    .Input("index_handle: resource")
    .Input("filename: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        return Status::OK();
    });

class InitializeIndexFromFileOp : public OpKernel {
   public:
    explicit InitializeIndexFromFileOp(OpKernelConstruction *ctx)
        : OpKernel(ctx) {}

    void Compute(OpKernelContext *ctx) override {
        mutex_lock l(mu_);
        Index *index;
        OP_REQUIRES_OK(ctx, GetIndex("index_handle", ctx, &index));
        core::ScopedUnref unref_me(index);

        DataType expected_input_0 =
            (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
        DataTypeVector expected_inputs = {expected_input_0, DT_STRING};
        DataTypeVector expected_outputs = {};
        OP_REQUIRES_OK(ctx,
                       ctx->MatchSignature(expected_inputs, expected_outputs));

        const Tensor &index_filename_tensor = ctx->input(1);
        OP_REQUIRES(ctx,
                    TensorShapeUtils::IsScalar(index_filename_tensor.shape()),
                    errors::InvalidArgument(
                        "filename should be a single string, but got ",
                        index_filename_tensor.shape().DebugString()));

        const string &index_filename =
            index_filename_tensor.scalar<tstring>()();
        OP_REQUIRES(ctx, !index_filename.empty(),
                    errors::InvalidArgument("filename cannot be empty."));

        int64 memory_used_before = 0;
        if (ctx->track_allocations()) {
            memory_used_before = index->MemoryUsed();
        }
        OP_REQUIRES_OK(ctx, index->InitializeFromFile(ctx, index_filename));
        if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(index->MemoryUsed() -
                                                     memory_used_before);
        }
    }

   private:
    mutex mu_;

    TF_DISALLOW_COPY_AND_ASSIGN(InitializeIndexFromFileOp);
};

REGISTER_OP("IndexSearch")
    .Input("index_handle: resource")
    .Input("query_embedding: float32")
    .Input("n_neighbor: int32")
    .Input("n_probe: int32")
    .Input("metric_type: int32")
    // .Attr("verbose: bool = true")
    .Output("neighbors: int64")
    .Output("scores: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class IndexSearchOp : public OpKernel {
   public:
    explicit IndexSearchOp(OpKernelConstruction *context) : OpKernel(context) {
        // OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
    }

    void Compute(OpKernelContext *context) override {
        Index *index;
        OP_REQUIRES_OK(context, GetIndex("index_handle", context, &index));
        core::ScopedUnref unref_me(index);

        // Grab the input tensor
        const Tensor &query_tensor = context->input(1);
        int n_neighbor = context->input(2).scalar<int>()(0);
        int n_probe = context->input(3).scalar<int>()(0);
        MetricType metric_type =
            static_cast<MetricType>(context->input(4).scalar<int>()(0));

        index->NNSearch(context, query_tensor, n_neighbor, n_probe, metric_type);
    }

   private:
    bool verbose_;
};


REGISTER_KERNEL_BUILDER(Name("InitializeIndexFromFile").Device(DEVICE_CPU),
                        InitializeIndexFromFileOp);

REGISTER_KERNEL_BUILDER(Name("IndexSearch").Device(DEVICE_CPU), IndexSearchOp);

REGISTER_KERNEL_BUILDER(Name("Index").Device(DEVICE_CPU), IndexOp);