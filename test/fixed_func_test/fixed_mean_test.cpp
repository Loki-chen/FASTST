#include "ezpc_scilib/ezpc_utils.h"
#include <model.h>
#include <cmath>
#define TEST

int main()
{
    // fixed-mean test
    int array_size = 3072;
    int len = 128;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    double *input = new double[len * array_size];
    int64_t *fix_input = new int64_t[len * array_size];
    uint64_t *unsig_fix_input = new uint64_t[len * array_size];
    // std::cout << "real int_input neg_mod \n";
    double *sum = new double[len];
    for (size_t j = 0; j < len; j++)
    {
        double row_sum = 0;
        for (size_t i = 0; i < array_size; i++)
        {
            input[j * array_size + i] = dist(gen);
            fix_input[j * array_size + i] = static_cast<int64_t>(input[j * array_size + i] * (1ULL << 13));           // (-5, 5)
            unsig_fix_input[j * array_size + i] = sci::neg_mod(fix_input[j * array_size + i], (int64_t)(1ULL << 37)); // (0, 10)
            // std::cout << input[j * array_size + i] << " ";
            // std::cout << fix_input[j * array_size + i] << " ";
            // std::cout << unsig_fix_input[j * array_size + i] << " ";
            // std::cout << "\n";
            row_sum += input[j * array_size + i];
        }
        sum[j] = row_sum;
        // std::cout << "true mean: " << sum[j] / 5 << " \n";
        // std::cout << "true_mean_fixarry: " << sci::neg_mod(static_cast<int64_t>((sum[j] / 5) * (1ULL << 13)), (int64_t)(1ULL << 37)) << " \n";
    }

    double *mu = new double[len * array_size];
    double *mu2 = new double[len * array_size];

    double *sum_mu2 = new double[len];
    for (size_t j = 0; j < len; j++)
    {
        double row_sum = 0;
        for (size_t i = 0; i < array_size; i++)
        {
            mu[j * array_size + i] = input[j * array_size + i] - sum[j] / array_size;
            mu2[j * array_size + i] = mu[j * array_size + i] * mu[j * array_size + i];
            row_sum += mu2[j * array_size + i];
        }
        sum_mu2[j] = row_sum;
    }

    for (size_t j = 0; j < len; j++)
    {
        // std::cout << "mu2: " << sum_mu2[j] << " ";
        // std::cout << "true_fixarry: " << sci::neg_mod(static_cast<int64_t>((sum_mu2[j]) * (1ULL << 12)), (int64_t)(1ULL << 37));
        // std::cout << "\n";
        // std::cout << "avg_mu2: " << sum_mu2[j] / array_size << " ";
        // std::cout << "agv_true_fixarry: " << sci::neg_mod(static_cast<int64_t>((sum_mu2[j] / array_size) * (1ULL << 12)), (int64_t)(1ULL << 37));
        // std::cout << "\n";
        std::cout << "delta: " << 1.0 / sqrt(sum_mu2[j] / array_size) << " ";
        std::cout << "delta_fixarry: " << sci::neg_mod(static_cast<int64_t>((1.0 / sqrt(sum_mu2[j] / array_size)) * (1ULL << 13)), (int64_t)(1ULL << 37));
        std::cout << "\n";
    }

    sci::OTPack *otpack;
    sci::IOPack *iopack;

    FPMath *fpmath = new FPMath(sci::PUBLIC, iopack, otpack);
    FixOp *fix = new FixOp(sci::PUBLIC, iopack, otpack);
    vector<FixArray> input_array;
    for (size_t i = 0; i < len; i++)
    {
        input_array.push_back(fix->input(sci::PUBLIC, array_size, &unsig_fix_input[i * array_size], true, 37, 13));
    }

    vector<FixArray> mean = fpmath->mean(input_array);

    // unsigned_val();

    vector<FixArray> out_array = fpmath->standard_deviation(input_array, mean);

    for (size_t i = 0; i < len; i++)
    {
        out_array[i].party = sci::PUBLIC;
        print_fix(out_array[i]);
    }

    delete[] input;
    delete[] fix_input;
    delete[] unsig_fix_input;
    delete[] mu;
    delete[] mu2;
    delete[] sum_mu2;
    delete fpmath;
    delete fix;
}
