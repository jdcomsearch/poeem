#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace custom_ops {

namespace {
static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
}

using std::string;
using tensorflow::data::ParseScalarArgument;


class FeatureDict {
 public:
  FeatureDict(Env* env, const string& feature_dict_file)
      : env_(env), feature_dict_file_(feature_dict_file) {}

  Status Init() {
    if (Empty()) {
      return Status::OK();
    }
    LOG(INFO) << feature_dict_file_ << ": feature table is initializing";
    std::unique_ptr<RandomAccessFile> dict_file;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(feature_dict_file_, &dict_file));
    std::unique_ptr<io::InputBuffer> input_buffer;
    input_buffer.reset(new io::InputBuffer(dict_file.get(), kInputBufferSize));

    
    string line;
    int64 key = 0;
    for (uint64_t i = 0;; ++i) {
      const auto& status = input_buffer->ReadLine(&line);
      if (status.ok()) {
        int index = line.find_first_of('\01', 0);  // comma delimited file
        if (index <= 0){
          if (strings::SafeStringToNumeric(line, &key)) {
            if (i == 0) {
              LOG(INFO) << "feature size: " << 0;
            }
            lookup_map_[key] = std::move(line);
            key_vec_.push_back(key);
          } else {
            LOG(WARNING) << line << " line can not be correctly parsed";
          }
        } else if (!strings::SafeStringToNumeric(line.substr(0, index), &key)) {
              LOG(WARNING) << line << " line can not be correctly parsed";
        } else {
          if (i == 0) {
            int feature_size = str_util::Split(line.substr(index + 1), '\t').size();
            LOG(INFO) << "feature size: " << feature_size;
          }
          lookup_map_[key] = std::move(line);
          key_vec_.push_back(key);
        }
      } else if (errors::IsOutOfRange(status)) {
        break;
      } else {
        LOG(WARNING) << "unexpected error when initializing feature table"
                     << status;
        return status;
      }
      if (i % 1000000 == 0 && i > 0) {
        LOG(INFO) << i  << " lines has been loaded";
      }
    }
    LOG(INFO) << feature_dict_file_ << ": " << lookup_map_.size()
              << " features has been loaded";
    return Status::OK();
  }

  const string& Get(int64 key) const { return lookup_map_.at(key); }
  bool Empty() const { return feature_dict_file_.size() == 0; }
  size_t Size() const { return lookup_map_.size(); }
  const string& RandValue() const {
    int rand_key_idx = rand() % key_vec_.size();
    int64 rand_key = key_vec_.at(rand_key_idx);
    return lookup_map_.at(rand_key);
  }

 private:
  Env* env_;
  const string feature_dict_file_;
  std::unordered_map<int64, string> lookup_map_;
  std::vector<int64> key_vec_;
};


class RecDatasetOp : public tensorflow::DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::DatasetBase** output) override {
    // get all input tensors
    const Tensor* train_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("train_file", &train_file_tensor));
    const Tensor* user_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("user_file", &user_file_tensor));
    const Tensor* item_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("item_file", &item_file_tensor));
    int user_column_index = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int>(ctx, "user_column_index",
                                                 &user_column_index));
    int item_column_index = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int>(ctx, "item_column_index",
                                                 &item_column_index));
    int neg_item_count = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int>(ctx, "neg_item_count", &neg_item_count));

    // sanity check
    OP_REQUIRES(ctx, train_file_tensor->dims() <= 1,
                errors::InvalidArgument("`train_file` must be a scalar or a vector."));
    OP_REQUIRES(ctx, user_file_tensor->dims() <= 1,
                errors::InvalidArgument("`user_file` must be a scalar or a vector."));
    OP_REQUIRES(ctx, item_file_tensor->dims() <= 1,
                errors::InvalidArgument("`user_file` must be a scalar or a vector."));

    OP_REQUIRES(ctx, user_column_index >= 0,
              errors::InvalidArgument(
                  "user_column_index should not be negative if user "
                  "feature dict is given"));
    OP_REQUIRES(ctx, item_column_index >= 0,
              errors::InvalidArgument(
                  "item_column_index should not be negative if "
                  "item feature dict is given"));
    OP_REQUIRES(ctx, neg_item_count >= 0,
              errors::InvalidArgument(
                  "neg_item_count should not be negative"));

    std::string train_file = train_file_tensor->flat<string>()(0);
    std::string user_file = user_file_tensor->flat<string>()(0);
    std::string item_file = item_file_tensor->flat<string>()(0);
    LOG(INFO) << "files: " << train_file << ", " << user_file << ", " << item_file;

    *output = new Dataset(ctx, train_file, user_file, item_file,
        user_column_index, item_column_index, neg_item_count);
  }

 private:
  class Dataset : public tensorflow::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx, 
            std::string train_file,
            std::string user_feature_dict_file,
            std::string item_feature_dict_file, 
            int user_column_index,
            int item_column_index, 
            int neg_item_count)
        : DatasetBase(DatasetContext(ctx)),
          env_(ctx->env()),
          train_file_(train_file),
          user_feature_dict_file_(user_feature_dict_file),
          item_feature_dict_file_(item_feature_dict_file),
          user_column_index_(user_column_index),
          item_column_index_(item_column_index),
          neg_item_count_(neg_item_count) {}

    std::unique_ptr<tensorflow::IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<tensorflow::IteratorBase>(
          new Iterator({this, tensorflow::strings::StrCat(prefix, "::Rec")}));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    const tensorflow::DataTypeVector& output_dtypes() const override {
      static auto* const dtypes = new tensorflow::DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "RecDatasetOp::Dataset"; }
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              tensorflow::Node** output) const override {
        Node* train_file = nullptr;
        Node* user_feature_dict_file = nullptr;
        Node* item_feature_dict_file = nullptr;
        Node* user_column_index = nullptr;
        Node* item_column_index = nullptr;
        Node* neg_item_count = nullptr;

        TF_RETURN_IF_ERROR(b->AddScalar(train_file_, &train_file));
        TF_RETURN_IF_ERROR(
            b->AddScalar(user_feature_dict_file_, &user_feature_dict_file));
        TF_RETURN_IF_ERROR(
            b->AddScalar(item_feature_dict_file_, &item_feature_dict_file));
        TF_RETURN_IF_ERROR(b->AddScalar(user_column_index_, &user_column_index));
        TF_RETURN_IF_ERROR(b->AddScalar(item_column_index_, &item_column_index));
        TF_RETURN_IF_ERROR(b->AddScalar(neg_item_count_, &neg_item_count));

        TF_RETURN_IF_ERROR(b->AddDataset(
            this,
            {train_file, user_feature_dict_file, item_feature_dict_file,
             user_column_index, item_column_index, neg_item_count},
            output));
        return Status::OK();
    }

   private:
    Env* env_;
    const std::string train_file_;
    const std::string user_feature_dict_file_;
    const std::string item_feature_dict_file_;
    int user_column_index_;
    int item_column_index_;
    int neg_item_count_;

    class Iterator : public tensorflow::DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            batch_stream_(nullptr),
            user_feature_dict_(dataset()->env_,
                               dataset()->user_feature_dict_file_),
            item_feature_dict_(dataset()->env_,
                               dataset()->item_feature_dict_file_) {
          auto status1 = user_feature_dict_.Init();
          auto status2 = item_feature_dict_.Init();
          LOG(INFO) << "user feature dictionary initialization done, size = " 
                    << user_feature_dict_.Size();
          LOG(INFO) << "item feature dictionary initialization done, size = "
                    << item_feature_dict_.Size();
          LOG(INFO) << dataset()->train_file_ << ": reading training file";
          auto status = dataset()->env_->NewRandomAccessFile(
              dataset()->train_file_, &file_);
          if (!status.ok()) {
              LOG(ERROR) << "init random access file error";
          }
          input_stream_.reset(
              new io::RandomAccessInputStream(file_.get(), false));
          batch_stream_.reset(new io::BufferedInputStream(
              input_stream_.get(), kInputBufferSize, false));
      }

      Status GetNextInternal(tensorflow::IteratorContext* ctx,
                             std::vector<tensorflow::Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        string line;
        Status status = ReadLine(&line, end_of_sequence);
        if (status.ok()) {
          if (*end_of_sequence == true) {
            return status;
          }

          out_tensors->emplace_back(DT_STRING, TensorShape({}));
          TF_RETURN_IF_ERROR(AssembleExampleContent(&line));
          out_tensors->back().scalar<string>()() = std::move(line);
          *end_of_sequence = false;
          return Status::OK();
        } else {
          return status;
        }
      }

     private:
      Status ReadLine(string* line, bool* end_of_sequence) {
          mutex_lock l(mu_);
          Status status = batch_stream_->ReadLine(line);
          if (errors::IsOutOfRange(status)) {
            *end_of_sequence = true;
            return Status::OK();
          } else {
            return status;
          }
      }

      Status AssembleExampleContent(string* line_content) const {
          if (user_feature_dict_.Empty() && item_feature_dict_.Empty()) {
              // no need to insert anything
              return Status::OK();
          }

          auto tokens = str_util::Split(*line_content, '\t');
          int64 key = 0;
          if (!user_feature_dict_.Empty() && dataset()->user_column_index_ >= 0) {
              strings::SafeStringToNumeric(
                  tokens[dataset()->user_column_index_], &key);
              tokens[dataset()->user_column_index_] =
                  user_feature_dict_.Get(key);
          }
          if (!item_feature_dict_.Empty() &&
              dataset()->item_column_index_ >= 0) {
              strings::SafeStringToNumeric(
                  tokens[dataset()->item_column_index_], &key);
              tokens[dataset()->item_column_index_] =
                  item_feature_dict_.Get(key);
          }
          // sample negative items
          std::vector<string> neg_items;
          for (int i = 0; i < dataset()->neg_item_count_; i++) {
              std::string rand_item = item_feature_dict_.RandValue();
              // std::replace(rand_item.begin(), rand_item.end(), ',', '|');
              neg_items.emplace_back(rand_item);
          }

          std::string neg_items_str = str_util::Join(neg_items, ":");
          tokens.emplace_back(neg_items_str);

          *line_content = str_util::Join(tokens, "\t");

          return Status::OK();
      }

      mutex mu_;
      std::unique_ptr<io::BufferedInputStream> batch_stream_ GUARDED_BY(mu_);
      std::unique_ptr<io::RandomAccessInputStream> input_stream_ GUARDED_BY(mu_);
      std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
      FeatureDict user_feature_dict_;
      FeatureDict item_feature_dict_;
    };
  };
};

REGISTER_OP("RecDataset")
    .Input("train_file: string")
    .Input("user_file: string")
    .Input("item_file: string")
    .Input("user_column_index: int32")
    .Input("item_column_index: int32")
    .Input("neg_item_count: int32")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return Status::OK();
    });

// Register the kernel implementation for KNNDataset.
REGISTER_KERNEL_BUILDER(Name("RecDataset").Device(tensorflow::DEVICE_CPU), RecDatasetOp);

}  // custom_ops
}  // tensorflow
