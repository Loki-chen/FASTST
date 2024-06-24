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
    // std::cout << "real int_input neg_mod \n";
    double sum;
    for (size_t i = 0; i < array_size; i++)
    {
        input[i] = dist(gen);
        fix_input[i] = static_cast<int64_t>(input[i] * (1ULL << 12));           // (-5, 5)
        unsig_fix_input[i] = sci::neg_mod(fix_input[i], (int64_t)(1ULL << 37)); // (0, 10)
        // std::cout << input[i] << " ";
        // std::cout << fix_input[i] << " ";
        // std::cout << unsig_fix_input[i] << " ";
        // std::cout << "\n";
        sum += input[i];
    }
    // std::cout << "true: " << sum / 5 << " \n";
    // std::cout << "true_ fixarry: " << sci::neg_mod(static_cast<int64_t>((sum / 5) * (1ULL << 12)), (int64_t)(1ULL << 37)) << " \n";

    double *mu = new double[array_size];
    double *mu2 = new double[array_size];
    double sum_mu2;

    for (size_t i = 0; i < array_size; i++)
    {
        mu[i] = input[i] - sum / 5;
        mu2[i] += mu[i] * mu[i];
        sum_mu2 += mu2[i];
    }
    std::cout << "mu2: " << sum_mu2 << " ";
    std::cout << "true_fixarry: " << sci::neg_mod(static_cast<int64_t>((sum_mu2) * (1ULL << 12)), (int64_t)(1ULL << 37));
    std::cout << "avg_mu2: " << sum_mu2 / array_size << " ";
    std::cout << "agv_true_fixarry: " << sci::neg_mod(static_cast<int64_t>((sum_mu2 / array_size) * (1ULL << 12)), (int64_t)(1ULL << 37));
    std::cout << "\n";
    sci::OTPack *otpack;
    sci::IOPack *iopack;

    FPMath *fpmath = new FPMath(sci::PUBLIC, iopack, otpack);
    FixOp *fix = new FixOp(sci::PUBLIC, iopack, otpack);
    vector<FixArray> input_array;
    for (size_t i = 0; i < len; i++)
    {
        input_array.push_back(fix->input(sci::PUBLIC, array_size, &unsig_fix_input[i * array_size], true, 37, 12));
    }

    vector<FixArray> mean = fpmath->mean(input_array);

    // unsigned_val();

    vector<FixArray>
        out_array = fpmath->standard_deviation(input_array, mean);
    std::cout << "test end \n";
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
