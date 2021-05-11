#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"


#include "knn_dataset_op.h"


namespace tensorflow {
namespace custom_ops {

namespace {
static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
static const int kThreadBufferSize = 10000000;
}

using tensorflow::data::ParseScalarArgument;

/////////////////////////////////////////////////////////////////////
///
///  This is feature dictionary class to manage user or item features
///
///  The file format needs to follow the below requirement
///  1. the first column should be user id or item id
///  2. the id is followed by a delimiter '\01'
///  3. the features are delimited by tab '\t'
///  4. the order of features must be the same as in the final assembled training file
///
/////////////////////////////////////////////////////////////////////

FeatureDict::FeatureDict(Env* env, const string& feature_dict_file)
    : env_(env), feature_dict_file_(feature_dict_file), default_value_("") {}

void FeatureDict::ThreadParse(std::shared_ptr<std::vector<std::string>> blocks, int tid) {
    std::unordered_map<int64, string> thread_lookup_map;
    std::vector<int64> thread_key_vec;
    int64 key = 0;
    for (std::string& line : *blocks) {
        int index = line.find_first_of('\01', 0);
        if (index <= 0 ||
            !strings::SafeStringToNumeric(line.substr(0, index), &key)) {
            LOG(WARNING) << line << " line can not be correctly parsed";
        } else {
            line[index] = '\t';
            thread_lookup_map[key] = std::move(line);
            thread_key_vec.push_back(key);
        }
    }
    // needs synchronization to write into global lookup_maps
    mtx_.lock();
    for (const auto& itor : thread_lookup_map) {
        lookup_map_[itor.first] = itor.second;
    }
    for (const auto& key : thread_key_vec) {
        key_vec_.push_back(key);
    }
    mtx_.unlock();
    LOG(INFO) << "feaure initializing, blocksize " << blocks->size()
              << ", thread " << tid << " end";
}

Status FeatureDict::Init() {
    // not thread safe, need to assume we have only one instance of this class.
    if (initialized_) {
        return Status::OK();
    }
    LOG(INFO) << feature_dict_file_ << ": feature table is initializing";
    std::unique_ptr<RandomAccessFile> dict_file;
    env_->NewRandomAccessFile(feature_dict_file_, &dict_file);
    std::unique_ptr<io::InputBuffer> input_buffer;
    input_buffer.reset(new io::InputBuffer(dict_file.get(), kInputBufferSize));

    std::vector<std::thread> threads;
    std::shared_ptr<std::vector<std::string>> buff =
        std::make_shared<std::vector<std::string>>();

    string line;
    for (uint64_t i = 0;; ++i) {
        const auto& status = input_buffer->ReadLine(&line);
        if (status.ok()) {
            if (i == 0) {
                int index = line.find_first_of('\01', 0);
                int feature_size =
                    str_util::Split(line.substr(index + 1), '\t').size();
                LOG(INFO) << "feature size: " << feature_size;
                std::vector<string> null_tokens(feature_size, "NULL");
                null_tokens.insert(
                    null_tokens.begin(),
                    "0");  // 0 is place holder id for empty negatives
                lookup_map_[0] = str_util::Join(null_tokens, "\t");
                default_value_ = str_util::Join(null_tokens, "\t");
            }
            buff->push_back(line);
            if (buff->size() >= kThreadBufferSize) {
                threads.emplace_back(&FeatureDict::ThreadParse, this, buff,
                                     threads.size());
                buff = std::make_shared<std::vector<std::string>>();
            }
        } else if (errors::IsOutOfRange(status)) {
            break;
        } else {
            LOG(WARNING) << "unexpected error when initializing feature table"
                         << status;
            return status;
        }
    }
    if (buff->size() > 0) {
        threads.emplace_back(&FeatureDict::ThreadParse, this, buff,
                             threads.size());
    }
    for (auto& th : threads) {
        th.join();
    }
    initialized_ = true;
    LOG(INFO) << feature_dict_file_ << ": " << lookup_map_.size()
              << " features has been loaded";
    return Status::OK();
}

const string& FeatureDict::Get(int64 key) const {
    // return lookup_map_.at(key);
    return gtl::FindWithDefault(lookup_map_, key, default_value_);
}

const string& FeatureDict::RandValue() const {
    int rand_key_indx = rand() % key_vec_.size();
    int64 rand_key = key_vec_.at(rand_key_indx);
    return lookup_map_.at(rand_key);
}

DatasetIteratorImpl::DatasetIteratorImpl(Env* env,
                                        const std::string& batch_file,
                                        const std::string& item_dict_file,
                                        int positive_item_column_index,
                                        int random_negative_item_count)
    : batch_stream_(nullptr),
      item_feature_dict_(env, item_dict_file),
      positive_item_column_index_(positive_item_column_index),
      random_negative_item_count_(random_negative_item_count) {
    item_feature_dict_.Init();
    auto status = env->NewRandomAccessFile(batch_file, &file_);
    if (!status.ok()) {
        LOG(ERROR) << "init random access file error";
    }
    input_stream_.reset(new io::RandomAccessInputStream(file_.get(), false));
    batch_stream_.reset(new io::BufferedInputStream(input_stream_.get(),
                                                    kInputBufferSize, false));
}

Status DatasetIteratorImpl::GetNext(std::string* out, bool* end_of_sequence) {
    // mutex_lock l(mu_);
    string line;
    Status status = ReadLine(&line, end_of_sequence);
    if (status.ok()) {
        if (*end_of_sequence == true) {
            return status;
        }

        AssembleExampleContent(item_feature_dict_, &line);
        *out = std::move(line);
    }
    return status;
}

Status DatasetIteratorImpl::AssembleExampleContent(
    const FeatureDict& item_feature_dict, string* line_content) const {
    auto tokens = str_util::Split(*line_content, '\t');
    int64 key = 0;

    // insert positive item contents
    if (positive_item_column_index_ >= 0) {
        strings::SafeStringToNumeric(tokens[positive_item_column_index_], &key);
        tokens[positive_item_column_index_] = item_feature_dict.Get(key);
    }

    // insert randomly selected negative item contents
    if (random_negative_item_count_ > 0) {
        std::vector<std::string> rand_neg_items;
        rand_neg_items.reserve(random_negative_item_count_);
        for (int i = 0; i < random_negative_item_count_; i++) {
            std::string rand_item = item_feature_dict.RandValue();
            rand_neg_items.emplace_back(rand_item);
        }
        std::string rand_neg_items_str = str_util::Join(rand_neg_items, "\t");
        tokens.emplace_back(rand_neg_items_str);
    }

    *line_content = str_util::Join(tokens, "\t");
    return Status::OK();
}

Status DatasetIteratorImpl::ReadLine(string* line, bool* end_of_sequence) {
    Status status = batch_stream_->ReadLine(line);
    if (errors::IsOutOfRange(status)) {
        *end_of_sequence = true;
        return Status::OK();
    } else {
        return status;
    }
}

class KNNDatasetOp : public tensorflow::DatasetOpKernel {
   public:
    KNNDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

    void MakeDataset(tensorflow::OpKernelContext* ctx,
                     tensorflow::DatasetBase** output) override {
        // get all input parameters
        string batch_file = ctx->input(0).scalar<string>()(0);
        string item_file = ctx->input(1).scalar<string>()(0);
        int64 positive_item_column_index = -1;
        int64 random_negative_item_count = 0;

        OP_REQUIRES_OK(
            ctx, ParseScalarArgument<int64>(ctx, "positive_item_column_index",
                                            &positive_item_column_index));
        OP_REQUIRES_OK(
            ctx, ParseScalarArgument<int64>(ctx, "random_negative_item_count",
                                            &random_negative_item_count));

        // make dataset
        *output =
            new Dataset(ctx, batch_file, item_file, positive_item_column_index,
                        random_negative_item_count);
    }

   private:
    class Dataset : public tensorflow::DatasetBase {
       public:
        Dataset(tensorflow::OpKernelContext* ctx, std::string batch_file,
                std::string item_file, int positive_item_column_index,
                int random_negative_item_count)
            : DatasetBase(DatasetContext(ctx)),
              env_(ctx->env()),
              batch_file_(batch_file),
              item_file_(item_file),
              positive_item_column_index_(positive_item_column_index),
              random_negative_item_count_(random_negative_item_count) {}

        std::unique_ptr<tensorflow::IteratorBase> MakeIteratorInternal(
            const string& prefix) const override {
            return std::unique_ptr<tensorflow::IteratorBase>(new Iterator(
                {this, tensorflow::strings::StrCat(prefix, "::DisKnn")}));
        }

        const tensorflow::DataTypeVector& output_dtypes() const override {
            static auto* const dtypes =
                new tensorflow::DataTypeVector({DT_STRING});
            return *dtypes;
        }

        const std::vector<PartialTensorShape>& output_shapes() const override {
            static std::vector<PartialTensorShape>* shapes =
                new std::vector<PartialTensorShape>({{}});
            return *shapes;
        }

        string DebugString() const override { return "KNNDatasetOp::Dataset"; }
        Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  tensorflow::Node** output) const override {
            Node* batch_file = nullptr;
            Node* item_file = nullptr;
            Node* positive_item_column_index = nullptr;
            Node* random_negative_item_count = nullptr;

            TF_RETURN_IF_ERROR(b->AddScalar(batch_file_, &batch_file));
            TF_RETURN_IF_ERROR(b->AddScalar(item_file_, &item_file));
            TF_RETURN_IF_ERROR(b->AddScalar(positive_item_column_index_,
                                            &positive_item_column_index));
            TF_RETURN_IF_ERROR(b->AddScalar(random_negative_item_count_,
                                           &random_negative_item_count));

            TF_RETURN_IF_ERROR(b->AddDataset(
                this,
                {batch_file, item_file, positive_item_column_index,
                 random_negative_item_count},
                output));
            return Status::OK();
        }

       private:
        Env* env_;
        const std::string batch_file_;
        const std::string item_file_;
        int positive_item_column_index_;
        int random_negative_item_count_;

        class Iterator : public tensorflow::DatasetIterator<Dataset> {
           public:
            explicit Iterator(const Params& params)
                : DatasetIterator<Dataset>(params),
                  iterator_impl_(dataset()->env_, dataset()->batch_file_,
                                 dataset()->item_file_,
                                 dataset()->positive_item_column_index_,
                                 dataset()->random_negative_item_count_) {
            }

            Status GetNextInternal(tensorflow::IteratorContext* ctx,
                                   std::vector<tensorflow::Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
                std::string out;
                Status s = iterator_impl_.GetNext(&out, end_of_sequence);
                out_tensors->emplace_back(DT_STRING, TensorShape({}));
                out_tensors->back().scalar<string>()() = std::move(out);
                return s;
            }

           private:
            DatasetIteratorImpl iterator_impl_;
        };
    };
};

REGISTER_OP("KNNDataset")
    .Input("batch_file: string")
    .Input("item_feature_dict_file: string")
    .Input("positive_item_column_index: int32")
    .Input("random_negative_item_count: int32")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

// Register the kernel implementation for KNNDataset.
REGISTER_KERNEL_BUILDER(Name("KNNDataset").Device(tensorflow::DEVICE_CPU),
                        KNNDatasetOp);

}  // namespace custom_ops
}  // namespace tensorflow
