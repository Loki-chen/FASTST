#include "utils.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

using namespace sci;
using namespace std;

#define MAX_THREADS 32

// extern IOPack *iopackArr[MAX_THREADS];
// extern OTPack *otpackArr[MAX_THREADS];
// extern FPMath *fpmath[MAX_THREADS];

class NonLinear
{
public:
    int party;
    int port;
    string address;

    IOPack *iopackArr[MAX_THREADS];
    OTPack *otpackArr[MAX_THREADS];
    FPMath *fpmath[MAX_THREADS];

    NonLinear();

    NonLinear(int party, string address, int port);
    ~NonLinear();

    void softmax(int nthreads, uint64_t *input, uint64_t *output, int dim, int array_size, int ell, int s);
    void softmax_iron(int nthreads, uint64_t *input, uint64_t *output, int dim, int array_size, int ell, int s);

    void layer_norm(int nthreads, uint64_t *input, uint64_t *output, uint64_t *weight, uint64_t *bias, int dim, int array_size, int ell, int s);

    void gelu(int nthreads, uint64_t *input, uint64_t *output, int size, int ell, int s);
    void gelu_iron(int nthreads, uint64_t *input, uint64_t *output, int size, int ell, int s);

    void tanh(int nthreads, uint64_t *input, uint64_t *output, int size, int ell, int s);
    void tanh_iron(int nthreads, uint64_t *input, uint64_t *output, int size, int ell, int s);

    void gt_p_sub(int nthreads, uint64_t *input, uint64_t p, uint64_t *output, int size, int ell, int s_in, int s_out);

    void n_matrix_mul(int nthreads, uint64_t *input_1, uint64_t *input_2, uint64_t *output, int n, int dim1, int dim2, int dim3, int ell, int s);

    void n_matrix_mul_iron(int nthreads, uint64_t *input_1, uint64_t *input_2, uint64_t *output, int n, int dim1, int dim2, int dim3, int ell, int s_in_1, int s_in_2, int s_out);

    void print_ss(uint64_t *input, int length, int ell, int s);

    void right_shift(int nthreads, uint64_t *input, int a, uint64_t *output, int size, int ell, int s);

    FixArray to_public(uint64_t *input, int length, int ell, int s);

    // softmax_iron(vector<FixArray> input, int nthreads);

    // vector<FixArray> layer_norm(vector<FixArray> input, int nthreads);
    // vector<FixArray> layer_norm_iron(vector<FixArray> input, int nthreads);

    // FixArray gelu(FixArray input, int nthreads);
    // FixArray gelu_iron(FixArray input, int nthreads);
    // FixArray tanh(FixArray input, int nthreads);
    // FixArray tanh_iron(FixArray input, int nthreads);

    // void non_linear_thread_vector(int tid, vector<FixArray>& input, vector<FixArray>& output, vector<FixArray> (FPMath::*)(const vector<FixArray>& x));
    // void non_linear_thread(int tid, uint64_t* input, uint64_t* output, int nops, FixArray (FPMath::*)(const FixArray& x));
};