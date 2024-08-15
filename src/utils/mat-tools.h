#ifndef FAST_MAT_TOOLS_H__
#define FAST_MAT_TOOLS_H__
#include "he-tools.h"
#include "he-bfv.h"
#include "omp.h"
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using std::ifstream;
using std::istringstream;
using std::vector;

typedef vector<double> matrix;
typedef vector<uint64_t> bfv_matrix;

matrix matmul(const matrix &mat1, const matrix &mat2, size_t dim1, size_t dim2, size_t dim3, bool trans = false);
void random_mat(matrix &mat, double min = -1., double max = 1., bool binomial = false);
void random_bfv_mat(bfv_matrix &mat);
void random_ell_mat(bfv_matrix &mat, int ell);
void random_modP_mat(bfv_matrix &mat, int plain_mod);

matrix zero_sum(size_t row, size_t column);
void load_mat(matrix &mat, string path);
void load_bfv_mat(bfv_matrix &mat, string path);
void normalization(matrix &A, size_t row, size_t column);
matrix mean(const matrix &input, size_t row, size_t column);
matrix standard_deviation(const matrix &input, const matrix means, size_t row, size_t column);
void print_mat(const matrix &A, size_t row, size_t column);
void print_all_mat(const matrix &A, size_t row, size_t column);
LongCiphertext *RFCP_encodeA(const matrix &A, CKKSKey *party, CKKSEncoder *encoder, size_t dim1, size_t dim2,
                             size_t dim3);
BFVLongCiphertext *RFCP_bfv_encodeA(const bfv_matrix &A, BFVKey *party, size_t dim1, size_t dim2, size_t dim3);
LongCiphertext RFCP_matmul(const LongCiphertext *A_secret, const matrix &B, size_t dim1, size_t dim2, size_t dim3,
                           CKKSEncoder *encoder, Evaluator *evaluator);
BFVLongCiphertext RFCP_bfv_matmul(const BFVLongCiphertext *A_secret, const bfv_matrix &B, size_t dim1, size_t dim2, size_t dim3, BFVParm *parm);

template <class T> inline void send_mat(sci::NetIO *io, const vector<T> *mat, bool count_comm = true) {
    io->send_data(mat->data(), mat->size() * sizeof(double), count_comm);
}

template <class T> inline void recv_mat(sci::NetIO *io, vector<T> *mat, bool count_comm = true) {
    io->recv_data(mat->data(), mat->size() * sizeof(double), count_comm);
}

#endif