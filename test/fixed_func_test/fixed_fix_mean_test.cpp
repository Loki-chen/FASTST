#include "ezpc_scilib/ezpc_utils.h"
#include <model.h>
#include <cmath>
#define TEST

int main()
{
    // fixed-mean test
    int array_size = 5;
    int len = 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    double *input = new double[array_size];
    int64_t *fix_input = new int64_t[array_size];
    uint64_t *unsig_fix_input = new uint64_t[array_size];
    std::cout << "real int_input neg_mod \n";
    for (size_t i = 0; i < array_size; i++)
    {
        input[i] = dist(gen);
        fix_input[i] = static_cast<int64_t>(input[i] * (1ULL << 12));           // (-5, 5)
        unsig_fix_input[i] = sci::neg_mod(fix_input[i], (int64_t)(1ULL << 37)); // (0, 10)
        std::cout << input[i] << " ";
        std::cout << fix_input[i] << " ";
        std::cout << unsig_fix_input[i] << " ";
        std::cout << "\n";
    }
    sci::OTPack *otpack;
    sci::IOPack *iopack;

    FPMath *fpmath = new FPMath(sci::PUBLIC, iopack, otpack);
    FixOp *fix = new FixOp(sci::PUBLIC, iopack, otpack);
    vector<FixArray> input_array;
    for (size_t i = 0; i < len; i++)
    {
        input_array.push_back(fix->input(sci::PUBLIC, array_size, &unsig_fix_input[i * array_size], true, 37, 12));
    }

    vector<FixArray> out_array = fpmath->mean(input_array);

    for (size_t i = 0; i < len; i++)
    {
        out_array[i].party = sci::PUBLIC;
        print_fix(out_array[i]);
    }

    delete[] input;
    delete[] fix_input;
    delete[] unsig_fix_input;
    delete fpmath;
    delete fix;
}
