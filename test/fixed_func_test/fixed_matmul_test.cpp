#include "Utils/constants.h"
#include "Utils/net_io_channel.h"
#include "utils.h"
#include "gmp.h"
#include "cmath"
#include "utils/mat-tools.h"
#include <cstdint>
#include <vector>

void random_mat(vector<vector<int64_t>> &mat, double min, double max, int scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);
    int dim1 = mat.size();
    int dim2 = mat[0].size();

// #pragma omp parallel for
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            mat[i][j] = static_cast<int64_t>(dist(gen) * pow(2, scale));
        }
    }
}

vector<vector<vector<int64_t>>> mat_matrix_mul(vector<vector<int64_t>> &A, vector<vector<int64_t>> &B)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();

    vector<vector<vector<int64_t>>> C(rowsA, vector<vector<int64_t>>(colsB, vector<int64_t>(colsA)));
    vector<vector<int64_t>> C_packed(rowsA, vector<int64_t>(colsB, 0));
#pragma omp parallel for
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            for (size_t k = 0; k < colsA; k++) {
// #pragma omp critical
                C[i][j][k] = A[i][k] * B[k][j] ;
                C_packed[i][j] += C[i][j][k];
            }
        }
    }
    return C;
}

void mul2add(int party, NetIO *io, void* data) {

}

vector<vector<vector<int64_t>>> matmul_test(int party, NetIO *io, vector<vector<int64_t>> &A, vector<vector<int64_t>> &B, int scale) {
    assert(A[0].size() == B.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1. / scale, 1);
    int dim1 = A.size(), dim2 = B.size(), dim3 = B[0].size();
    if (party == sci::ALICE) {
        vector<vector<int64_t>> R(dim1, vector<int64_t>(dim2));
        vector<vector<int64_t>> Ae(dim1, vector<int64_t>(dim2));
#pragma omp parallel for
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                R[i][j] = static_cast<int64_t>(dist(gen) * pow(2, scale));
                Ae[i][j] = A[i][j] * R[i][j];
                Ae[i][j] /= static_cast<int64_t>(pow(2, scale));
            }
        }
        for (int i = 0; i < dim1; i++) {
            io->send_data(Ae[i].data(), dim2);
        }
        vector<vector<vector<int64_t>>> C(dim1, vector<vector<int64_t>>(dim3, vector<int64_t>(dim2)));
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < dim3; j++) {
                io->recv_data(C[i][j].data(), dim2);
            }
        }
#pragma omp parallel for
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim3; ++j) {
                for (size_t k = 0; k < dim2; ++k) {
                    C[i][j][k] /= R[i][k];
                }
            }
        }
        return C;
    } else {
        vector<vector<int64_t>> Ae(dim1, vector<int64_t>(dim2));
        for (int i = 0; i < dim1; i++) {
            io->recv_data(Ae[i].data(), dim2);
        }
        vector<vector<vector<int64_t>>> C(dim1, vector<vector<int64_t>>(dim3, vector<int64_t>(dim2))), 
                                        C_self(dim1, vector<vector<int64_t>>(dim3, vector<int64_t>(dim2)));
#pragma omp parallel for
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim3; ++j) {
                for (size_t k = 0; k < dim2; ++k) {
                    C_self[i][j][k] = static_cast<int64_t>(dist(gen) * pow(2, 2 * scale));
                    C[i][j][k] = Ae[i][k] * B[k][j];
                }
            }
        }
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim3; ++j) {
                io->send_data(C[i][j].data(), dim2);
            }
        }
    }
    return vector<vector<vector<int64_t>>>();
}

int main(int argc, const char** argv) {
    int dim1 = 128, dim2 = 768, dim3 = 64;
    int64_t field = 4294967311, scale = 12;
    INIT_TIMER
    if (argc > 1) {
        int party = argv[1][0] - '0';
        assert(party == sci::ALICE || party == sci::BOB);
        if (party == sci::ALICE){ std::cout << "Party: ALICE\n"; }
        else if (party == sci::BOB){ std::cout << "Party: BOB\n"; }
        sci::IOPack *iopack = new sci::IOPack(party, 56789);
        vector<vector<int64_t>> A(dim1, vector<int64_t>(dim2)), B(dim2, vector<int64_t>(dim3));
        random_mat(A, -1, 1, scale);
        random_mat(B, -1, 1, scale);
        size_t start = iopack->get_comm();
        START_TIMER
        auto C = matmul_test(party, iopack->io, A, B, scale);
        STOP_TIMER("matmul_test")
        std::cout << "comm: " << iopack->get_comm() - start << "\n";
        if (party == ALICE) {
            for (int i = 0; i < dim2; i++) {
                iopack->io->recv_data(B[i].data(), dim3);
            }
            START_TIMER
            auto C_true = mat_matrix_mul(A, B);
            STOP_TIMER("matmul")
            for (size_t i = 0; i < dim1; ++i) {
                for (size_t j = 0; j < dim3; ++j) {
                    for (size_t k = 0; k < dim2; ++k) {
                        C_true[i][j][k] /= static_cast<int64_t>(pow(2, scale));
                        C_true[i][j][k] -= C[i][j][k];
                    }
                }
            }
            std::cout << C_true[0][0][0] << " " << C_true[1][1][1] << " " << C_true[2][2][2] << "\n";
        } else {
            for (int i = 0; i < dim2; i++) {
                iopack->io->send_data(B[i].data(), dim3);
            }
        }
        
    }
    else { std::cout << "no party input\n"; }
    return 0;
}
