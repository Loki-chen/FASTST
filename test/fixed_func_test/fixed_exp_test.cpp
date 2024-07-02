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
    double *input = new double[len * array_size];
    int64_t *fix_input = new int64_t[len * array_size];
    uint64_t *unsig_fix_input = new uint64_t[len * array_size];

    std::cout << "real input: ";
    for (size_t j = 0; j < len; j++)
    {
        for (size_t i = 0; i < array_size; i++)
        {
            input[j * array_size + i] = dist(gen);
            fix_input[j * array_size + i] = static_cast<int64_t>(input[j * array_size + i] * (1ULL << 13));           // (-5, 5)
            unsig_fix_input[j * array_size + i] = sci::neg_mod(fix_input[j * array_size + i], (int64_t)(1ULL << 37)); // (0, 10)
            std::cout << input[j * array_size + i] << " ";
        }
    }

    std::cout << "\n";
    sci::OTPack *otpack;
    sci::IOPack *iopack;

    FPMath *fpmath = new FPMath(sci::PUBLIC, iopack, otpack);
    FixOp *fix = new FixOp(sci::PUBLIC, iopack, otpack);
    BoolOp *bool_op = new BoolOp(sci::PUBLIC, iopack, otpack);
    FixArray input_array = fix->input(sci::PUBLIC, len * array_size, unsig_fix_input, true, 37, 13);

    FixArray ret = fpmath->location_exp(input_array, input_array.s, input_array.s);

    std::cout << "compute exp: ";
    for (size_t j = 0; j < len * array_size; j++)
    {
        std::cout << double(sci::signed_val(ret.data[j], ret.ell)) / (1ULL << 13) << " ";
    }
    std::cout << "\n";

    std::cout << "true exp: ";
    for (size_t j = 0; j < len * array_size; j++)
    {
        std::cout << std::exp(input[j]) << " ";
    }
    std::cout << "\n";
    delete[] input;
    delete[] fix_input;
    delete[] unsig_fix_input;

    delete fpmath;
    delete fix;
}
