#include <iostream>
#include <string>
#include <vector>
#include "../cpp/vector_util.h"

typedef void (*VecFuncPtr)(const float *x, const float *y, size_t d, float *out);

void TestVectorFunc(std::string desc, VecFuncPtr Func1, VecFuncPtr Func2) {
    const float x[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    const float y[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
    size_t len = sizeof(x) / sizeof(float);

    int error_count = 0;
    for (int d = 1; d <= len; ++d) {
        std::vector<float> out1(d, -1.0);
        std::vector<float> out2(d, -1.0);
        (*Func1)(x, y, d, out1.data());
        (*Func2)(x, y, d, out2.data());

        if (out1 != out2) {
            ++error_count;
            std::cerr << "Error: " << desc << ", d=" << d << std::endl;
        }
    }

    if (error_count == 0) {
        std::cout << desc << " checking OK!" << std::endl;
    }
}

int main() {
    TestVectorFunc("Vector Addition", simple_fvec_addition, fvec_addition);
    TestVectorFunc("Vector Substraction", simple_fvec_substraction, fvec_substraction);

    return 0;
}
