#include <bits/stdc++.h>
using std::vector;

std::vector<double> matmul(std::vector<double> &mat1, std::vector<double> &mat2, size_t dim1, size_t dim2, size_t dim3, bool trans = false);
void random_mat(std::vector<double> &mat);
std::vector<double> zero_sum(size_t row, size_t column);
void norm(std::vector<double> &A);
void print_mat(std::vector<double> A, size_t dim1, size_t dim2);