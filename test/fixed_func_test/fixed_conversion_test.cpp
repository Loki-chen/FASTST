#include <iostream>
#include <vector>
#include "ezpc_scilib/ezpc_utils.h"
#define TEST
uint64_t *Ring_to_Prime(uint64_t *input, int length, int ell, int plain_mod)
{
#ifdef LOG
    auto t_conversion = high_resolution_clock::now();
#endif

    uint64_t *output = new uint64_t[length];
    vector<uint64_t> tmp;
    for (size_t i = 0; i < length; i++)
    {
        tmp[i] = sci::neg_mod(sci::signed_val(input[i], ell), (int64_t)plain_mod);
    }
    memcpy(output, tmp.data(), length * sizeof(uint64_t));
#ifdef LOG
    t_total_conversion += interval(t_conversion);
#endif
    return output;
}

int main()
{
    int array_size = 5;
    int len = 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    double *input = new double[len * array_size];
    int64_t *fix_input = new int64_t[len * array_size];
    uint64_t *unsig_fix_input = new uint64_t[len * array_size];

    for (size_t j = 0; j < len; j++)
    {
        for (size_t i = 0; i < array_size; i++)
        {
            input[j * array_size + i] = dist(gen);
            fix_input[j * array_size + i] = static_cast<int64_t>(input[j * array_size + i] * (1ULL << 13));           // (-5, 5)
            unsig_fix_input[j * array_size + i] = sci::neg_mod(fix_input[j * array_size + i], (int64_t)(1ULL << 37)); // (0, 10)
            // std::cout << input[j * array_size + i] << " ";
            // std::cout << fix_input[j * array_size + i] << " ";
            // std::cout << unsig_fix_input[j * array_size + i] << " ";
            // std::cout << "\n";
        }
    }
    uint64_t *output = new uint64_t[len * array_size];

    delete[] input;
    delete[] fix_input;
    delete[] unsig_fix_input;
    delete[] output;

    return 0;
}