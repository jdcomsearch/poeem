#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>
#include <algorithm>
#include <locale.h>
#include "tensorflow/core/platform/types.h"
#include "absl/strings/str_split.h"

namespace tensorflow {

typedef int (*TokenizerFn)(const std::string &, std::vector<std::string> *);

constexpr const char* HEAD = "^ ";
constexpr const char* TAIL = " $";

constexpr const char* HC = "^";
constexpr const char* SPACE = " ";
constexpr const char* TC = "$";

REGISTER_OP("UnigramsAndEnTrigramParser")
    .Input("query: string")
    .Output("token_indices: int64")
    .Output("token_values: string")
    .Output("token_shapes: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
          return Status::OK();
    });

REGISTER_OP("BigramsAndEnTrigramParser")
    .Input("query: string")
    .Output("token_indices: int64")
    .Output("token_values: string")
    .Output("token_shapes: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
          return Status::OK();
    });

inline wchar_t to_wchar(const char *src, int len, std::mbstate_t *state) {
    wchar_t wch = L'\0';
    static bool locale_not_set = true;
    if (locale_not_set) { // Last resort locale setter.
      setlocale(LC_ALL, "en_US.utf8");
      locale_not_set = false;
    }
    int iLen = std::mbrtowc(&wch, src, len, state);
    if (iLen <= 0) {
        wch = L'\0';
    }
    return wch;
}

inline int n_octets(unsigned char lb) {
    if (( lb & 0x80 ) == 0 )          // lead bit is zero, must be a single ascii
        return 1;
    else if (( lb & 0xE0 ) == 0xC0 )  // 110x xxxx
        return 2;
    else if (( lb & 0xF0 ) == 0xE0 ) // 1110 xxxx
        return 3;
    else if (( lb & 0xF8 ) == 0xF0 ) // 1111 0xxx
        return 4;
    else
        return -1;
}

int get_alpha_trigrams(const std::string &str, std::vector<std::string>* out) {
    if (str.empty()) {
        return 0;
    }

    std::string str_extend = "#" + str + "#";
    for (std::size_t i = 0; i < str_extend.size() - 2; i++) {
        std::string letter_tri_gram = str_extend.substr(i, 3);
        out->emplace_back(letter_tri_gram);
    }

    return 0;
}

int get_alphanum(const char *pBegin,
                 const char *pEnd,
                 std::string* word) {
    auto is_delim = [](char ch)->bool {
       // 只保留字母和数字
       return !isalpha(ch) && !isdigit(ch);
    };
    const char *pCur = pBegin;
    // skip invalid chars
    while (pCur < pEnd && n_octets(*pCur) == 1 && is_delim(*pCur)) {
      ++pCur;
    }
    const char *pLast = pCur;
    // find end of a word
    while (pCur < pEnd && n_octets(*pCur) == 1 && !is_delim(*pCur)) {
      ++pCur;
    }
    auto len = pCur - pLast;
    *word = std::string(pLast, len);
    std::transform(word->begin(), word->end(), word->begin(), ::tolower);

    auto step = pCur - pBegin;  // return the step that we need to move pCur
    return step;
}

int get_unigrams(const std::string &str,
                 std::vector<std::string>* out,
                 std::vector<std::string>* alpha_tokens,
                 const bool is_alpha_trigrams,
                 const bool is_alpha_included) {
    const char *pCur = str.c_str();
    const char *pEnd = str.c_str() + str.length();
    std::mbstate_t state = std::mbstate_t();

    out->reserve(str.length() / 3 + 1);
    alpha_tokens->reserve(3);
    std::string word;
    while (pCur < pEnd) {
        int n = n_octets(*pCur);
        if (n > 1) { // chinese char
            wchar_t wchar = to_wchar(pCur, n, &state);
            if (wchar >= L'\u4e00' && wchar <= L'\u9fff') {
                out->emplace_back(pCur, n);
            }
            pCur += n;
        } else if (n == 1) { // ascii char
            int step = get_alphanum(pCur, pEnd, &word);
            pCur += step;
            if (word.size() == 0) {
              continue;
            }

            if (is_alpha_included) {
                out->push_back(word);
            }
            if (is_alpha_trigrams) {
                get_alpha_trigrams(word, out);
            }
            if (word.length() > 3) {  // not included in trigrams
                alpha_tokens->push_back(word);
            }
        } else {    // n_octets fail
            ++pCur;
        }
    } // while

    return 0;
}

int get_bigrams(const std::vector<std::string>& tokens,
                std::vector<std::string>* out) {
    if (tokens.empty()) {
        out->emplace_back(HC + std::string(SPACE) + TC);
        return 0;
    }
    auto length = tokens.size();
    // set first elem
    out->emplace_back(HEAD + tokens[0]);
    // set mid elems
    for (size_t i = 1; i < length; ++i) {
        out->emplace_back(tokens[i - 1] + SPACE + tokens[i]);
    }
    // set last one
    out->emplace_back(tokens[length - 1] + TAIL);
    return 0;
}

int str_preprocess(const std::string &inp, std::string* out) {
    (*out) = inp;
    if (!out->empty()) {
        out->erase(0, out->find_first_not_of(" \t\n"));
        out->erase(out->find_last_not_of(" \t\n") + 1);
    }
    return 0;
}

int unigrams_and_en_trigram_parser(const std::string &str,
                                   std::vector<std::string>* out) {
    std::string inputstr;
    str_preprocess(str, &inputstr);

    std::vector<std::string> alpha_tokens;
    get_unigrams(inputstr, out, &alpha_tokens, true, false);

    return 0;
}

int bigrams_and_en_trigram_parser(const std::string &str,
                                  std::vector<std::string>* out) {
    std::string inputstr;
    str_preprocess(str, &inputstr);

    std::vector<std::string> tokens;
    std::vector<std::string> alpha_tokens;
    get_unigrams(inputstr, &tokens, &alpha_tokens, true, false);

    std::vector<std::string> bigrams;
    get_bigrams(tokens, &bigrams);

    out->reserve(tokens.size() + bigrams.size());
    out->insert(out->end(), tokens.begin(), tokens.end());
    out->insert(out->end(), bigrams.begin(), bigrams.end());

    return 0;
}

template<TokenizerFn tokenizer_knn>
class TokenizeSparseBaseKnnOp : public OpKernel {
public:
    explicit TokenizeSparseBaseKnnOp(OpKernelConstruction* context) :
      OpKernel(context) {
        setlocale(LC_ALL, "en_US.utf8");
    }

    void Compute(OpKernelContext* context) override {
        Tensor* indices_tensor = NULL;
        Tensor* values_tensor = NULL;
        Tensor* shapes_tensor = NULL;
        const Tensor& input_tensor = context->input(0);

        auto input = input_tensor.flat<string>();
        const size_t N = input.size();
        std::vector<string> values;
        std::vector<std::pair<int64,int64>> indexes;

        int64 max_size = 0;
        for (size_t i = 0; i < N; i++) {
            std::vector<string> out;
            tokenizer_knn(input(i), &out);
            for (size_t j = 0; j < out.size(); ++j) {
                values.push_back(std::move(out[j]));
                indexes.emplace_back(i, j);
            }
            max_size = std::max(max_size, static_cast<int64>(out.size()));
        }
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape{static_cast<int64>(values.size()), 2},
                &indices_tensor));  // 2}, &indices_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(
            1, TensorShape{static_cast<int64>(values.size())}, &values_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(
            2, TensorShape{2}, &shapes_tensor));

        auto output_indices = indices_tensor->matrix<int64>();
        auto output_values = values_tensor->flat<string>();
        auto output_shapes = shapes_tensor->flat<int64>();
        output_shapes(0) = input.size();
        output_shapes(1) = max_size;

        for (size_t i = 0; i < values.size(); ++i) {
          output_indices(i, 0) = indexes[i].first;
          output_indices(i, 1) = indexes[i].second;
          output_values(i) = std::move(values[i]);
        }
    }
};

class BigramsAndEnTrigramParserOp : public TokenizeSparseBaseKnnOp<bigrams_and_en_trigram_parser>
{
    using TokenizeSparseBaseKnnOp::TokenizeSparseBaseKnnOp;
};
REGISTER_KERNEL_BUILDER(Name("BigramsAndEnTrigramParser").Device(DEVICE_CPU), BigramsAndEnTrigramParserOp);
class UnigramsAndEnTrigramParserOp : public TokenizeSparseBaseKnnOp<unigrams_and_en_trigram_parser>
{
    using TokenizeSparseBaseKnnOp::TokenizeSparseBaseKnnOp;
};
REGISTER_KERNEL_BUILDER(Name("UnigramsAndEnTrigramParser").Device(DEVICE_CPU), UnigramsAndEnTrigramParserOp);

}   // namespace tensorflow
