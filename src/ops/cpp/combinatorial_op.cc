#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <algorithm>
#include <queue>


namespace tensorflow {

namespace {

struct DistancePair {
  DistancePair(int i1, int i2, double d) : index1(i1), index2(i2), dist(d) {}

  bool operator<(const DistancePair& b1) const { return b1.dist < dist; }

  int index1, index2;
  float dist;
};

}  // namespace

REGISTER_OP("UndirectedBipartiteMatch")
    .Input("distance_mat: float")
    .Input("input_num_valid_rows: float")
    .Output("match_indices: int32")
    .Attr("top_k: int = -1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        c->set_output(0, c->MakeShape({c->Dim(input, 0)}));
        return Status::OK();
    });

class UndirectedBipartiteMatchOp : public OpKernel {
 public:
  explicit UndirectedBipartiteMatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("top_k", &top_k_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_distance_mat = context->input(0);
    OP_REQUIRES(context, input_distance_mat.dims() == 2,
                errors::InvalidArgument(
                    "distance_mat should be 2-dimensional, but got ",
                    input_distance_mat.shape().DebugString()));
    const int num_input_rows = input_distance_mat.dim_size(0);
    const int num_input_columns = input_distance_mat.dim_size(1);

    const Tensor& input_num_valid_rows = context->input(1);
    OP_REQUIRES(
        context, input_num_valid_rows.NumElements() == 1,
        errors::InvalidArgument(
            "num_valid_rows argument should be a tensor with 1 element, "
            "but got ",
            input_num_valid_rows.NumElements()));

    const float num_valid_rows_f = input_num_valid_rows.flat<float>()(0);
    int num_valid_rows = num_input_rows;
    // If num_valid_rows_f is non-negative, use it to set num_valid_rows.
    if (num_valid_rows_f >= 0) {
      num_valid_rows = static_cast<int>(num_valid_rows_f + 0.1);
    }
    OP_REQUIRES(
        context, num_input_rows >= num_valid_rows,
        errors::InvalidArgument("There should be at least ", num_valid_rows,
                                " rows in distance_mat, but only got ",
                                num_input_rows, " rows."));

    // If negative or zero then set it to the maximum possible matches.
    auto valid_top_k = top_k_;

    if (valid_top_k <= 0) {
      valid_top_k = num_valid_rows * num_input_columns;
    }

    // Create output tensors.
    Tensor* match_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_input_rows}),
                                            &match_indices));

    TTypes<float, 2>::ConstTensor distance_mat =
        input_distance_mat.shaped<float, 2>(
            {num_input_rows, num_input_columns});

    // Greedy bi-partite matching.
    std::priority_queue<DistancePair> match_queue;

    for (int index1 = 0; index1 < num_valid_rows; index1++) {
      for (int index2 = 0; index2 < num_input_columns; index2++) {
        match_queue.push(
            DistancePair(index1, index2, distance_mat(index1, index2)));
      }
    }

    std::vector<int> match_vec(num_input_rows, -1);
    int index = 0;
    while (!match_queue.empty()) {
      const auto& match = match_queue.top();
      if (match_vec[match.index1] == -1 &&
          match_vec[match.index2] == -1) {
        match_vec[match.index1] = match.index2;
        match_vec[match.index2] = match.index1;

        index++;
        if (index >= valid_top_k) {
          break;
        }
      }
      match_queue.pop();
    }

    // Set the output tensors.
    match_indices->vec<int>() =
        TTypes<int>::Vec(match_vec.data(), num_input_rows);
  }

 private:
  int top_k_;
};

REGISTER_KERNEL_BUILDER(Name("UndirectedBipartiteMatch").Device(DEVICE_CPU),
                        UndirectedBipartiteMatchOp);

} // namespace tensorflow

