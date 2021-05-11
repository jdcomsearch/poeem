
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace tensorflow {
namespace custom_ops {

/////////////////////////////////////////////////////////////////////
///
///  This is feature dictionary class to manage user or item features
///
///  The file format needs to follow the below requirement
///  1. the first column should be user id or item id
///  2. the id is followed by a delimiter '\01'
///  3. the features are delimited by tab '\t'
///  4. the order of features must be the same as in the final assembled
///  training file
///
/////////////////////////////////////////////////////////////////////
class FeatureDict {
   public:
    FeatureDict(Env *env, const std::string &feature_dict_file);

    FeatureDict(const FeatureDict &) = delete;
    FeatureDict &operator=(const FeatureDict &) = delete;
    FeatureDict(FeatureDict &&) = default;
    FeatureDict &operator=(FeatureDict &&) = default;

    Status Init();
    const string &Get(int64 key) const;
    const string &RandValue() const;

   private:
    void ThreadParse(std::shared_ptr<std::vector<std::string>> blocks, int tid);

    Env *env_;
    const std::string feature_dict_file_;
    std::string default_value_;
    static std::unordered_map<int64, std::string> lookup_map_;
    static std::vector<int64> key_vec_;
    static bool initialized_;
    std::mutex mtx_;
};

// static variables need to be initialized outside class definition
std::unordered_map<int64, std::string> FeatureDict::lookup_map_;
std::vector<int64> FeatureDict::key_vec_;
bool FeatureDict::initialized_ = false;


class DatasetIteratorImpl {
   public:
    DatasetIteratorImpl(Env *env, const std::string &batch_file,
                        const std::string &item_dict_file,
                        int positive_item_column_index,
                        int random_negative_item_count);

    Status GetNext(std::string *out, bool *end_of_sequence);

   private:
    Status AssembleExampleContent(const FeatureDict &item_feature_dict,
                                  string *line_content) const;

    Status ReadLine(string *line, bool *end_of_sequence);

    mutex mu_;
    std::unique_ptr<io::BufferedInputStream> batch_stream_ GUARDED_BY(mu_);
    std::unique_ptr<io::RandomAccessInputStream> input_stream_ GUARDED_BY(mu_);
    std::unique_ptr<RandomAccessFile> file_ GUARDED_BY(mu_);
    FeatureDict item_feature_dict_;
    int positive_item_column_index_;
    int random_negative_item_count_;
};

}  // namespace custom_ops
}  // namespace tensorflow