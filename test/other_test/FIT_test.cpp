#include <utils.h>

bfv_matrix matmul(const bfv_matrix &mat1, const bfv_matrix &mat2, size_t dim1, size_t dim2, size_t dim3, bool trans=false) {
    bfv_matrix result(dim1 * dim3);
    if (!trans) {
        {
#pragma omp parallel for
            for (size_t i = 0; i < dim1; i++) {
                const size_t base_idx1 = i * dim2;
                const size_t base_idx2 = i * dim3;
                for (size_t k = 0; k < dim2; k++) {
                    const size_t base_idx3 = k * dim3;
                    const uint64_t tmp = mat1[base_idx1 + k];
                    for (size_t j = 0; j < dim3; j++) {
                        result[base_idx2 + j] += tmp * mat2[base_idx3 + j];
                    }
                }
            }
        }
    } else {
        {
#pragma omp parallel for
            for (size_t i = 0; i < dim1; i++) {
                const size_t base_idx1 = i * dim2;
                const size_t base_idx2 = i * dim3;
                for (size_t j = 0; j < dim3; j++) {
                    const size_t base_idx3 = j * dim2;
                    uint64_t sum = 0.;
                    for (size_t k = 0; k < dim2; k++) {
                        sum += mat1[base_idx1 + k] * mat2[base_idx3 + k];
                    }
                    result[base_idx2 + j] = sum;
                }
            }
        }
    }
    return result;
}

int dim[5][3] = {
    {32, 32, 64},
    {128, 64, 64},
    {128, 256, 64},
    {128, 256, 256},
    {128, 768, 64}
};

int main(int argc, const char** argv) {
    if (argc > 2) {
		int party_ = argv[1][0] - '0', d = argv[2][0] - '0';
		assert(party_ == sci::ALICE || party_ == sci::BOB);
		if (party_ == sci::ALICE) std::cout << "Party: ALICE\n"; else std::cout << "Party: BOB\n";
        assert(0 <= d && d < 5);
        int dim1 = dim[d][0], dim2 = dim[d][1], dim3 = dim[d][2];
        std::cout << "dim1 = " << dim1 << ", dim2 = " << dim2 << ", dim3 = " << dim3 << "\n";
        BFVParm *parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
	    BFVKey *party = new BFVKey(party_, parm);
        string ip = "127.0.0.1";
        if (argc > 3) {
            ip = argv[3];
        }
        sci::IOPack *iopack = new sci::IOPack(party_, 56789, ip);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        sci::NetIO *io = iopack->io;
        bfv_matrix A(dim1 * dim2, 2), B(dim2 * dim3, 2);
        random_ell_mat(A, 12);
        random_ell_mat(B, 12);
        size_t start = iopack->get_comm();
        if (party_ == sci::ALICE) {
            bfv_matrix Ae(dim1 * dim2);
#pragma omp parallel for
            for (int i = 0; i < dim1; i++) {
                for (int j = 0; j < dim2; j++) {
                    Ae[j * dim1 + i] = A[i * dim2 + j];
                }
            }
            BFVLongCiphertext* Ae_sec = new BFVLongCiphertext[dim2 / 2];
#pragma omp parallel for
            for (int i = 0; i < dim2 / 2; i++) {
                BFVLongPlaintext lpt(parm, vector<uint64_t>(Ae.begin() + i * dim1 * 2, Ae.begin() + (i + 1) * dim1 * 2));
                Ae_sec[i] = BFVLongCiphertext(lpt, party);
            }
            INIT_TIMER
            START_TIMER
            for (int i = 0; i  < dim2 / 2; i++) {
                BFVLongCiphertext::send(io, Ae_sec + i);
            }
            delete[] Ae_sec;
{
    std::stringstream os;
    party->galois_keys.save(os);
    uint64_t key_size = os.tellp();
    string key_ser = os.str();
    io->send_data(&key_size, sizeof(uint64_t));
    io->send_data(key_ser.c_str(), key_ser.size());
}
            BFVLongCiphertext *C_sec = new BFVLongCiphertext[dim3];
            for (int i = 0; i < dim3; i++) {
                BFVLongCiphertext::recv(io, C_sec + i, parm->context);
            }
            vector<vector<uint64_t>> C(dim3);
#pragma omp parallel for
            for (int i = 0; i < dim3; i++) {
                auto C_p = C_sec[i].decrypt(party);
                auto C_vec = C_p.decode_uint(parm);
                C[i] = vector<uint64_t>(C_vec.begin(), C_vec.begin() + dim1);
            }
            STOP_TIMER("FCHE")
        } else {
            vector<vector<bfv_matrix>> Be(dim3, vector<bfv_matrix>(dim2 / 2, bfv_matrix(dim1 * 2)));
#pragma omp parallel for
            for (int i = 0; i < dim3; i++) {
                for (int j = 0; j < dim2 / 2; j++) {
                    for (int k = 0; k < dim1 * 2; k++) {
                        Be[i][j][k] = B[(2 * j + k / dim1) * dim3 + i];
                    }
                }
            }
            INIT_TIMER
            START_TIMER
            BFVLongCiphertext* Ae_sec = new BFVLongCiphertext[dim2 / 2], *C_sec = new BFVLongCiphertext[dim3];
            for (int i = 0; i  < dim2 / 2; i++) {
                BFVLongCiphertext::recv(io, Ae_sec + i, parm->context);
            }
            GaloisKeys alice_gk;
{
    std::stringstream is;
    uint64_t key_size;
    io->recv_data(&key_size, sizeof(uint64_t));
    char *key_result = new char[key_size];
    io->recv_data(key_result, key_size);
    is.write(key_result, key_size);
    alice_gk.unsafe_load(*(parm->context), is);
    delete[] key_result;
}
#pragma omp parallel for
            for (int i = 0; i < dim3; i++) {
                BFVLongCiphertext* Ae_sec_ = new BFVLongCiphertext[dim2 / 2];
                for (int j = 0; j < dim2 / 2; j++) {
                    BFVLongPlaintext Be_plain_j = BFVLongPlaintext(parm, Be[i][j]);
                    Ae_sec_[j] = Ae_sec[j].multiply_plain(Be_plain_j, parm->evaluator);
                }
#pragma omp critical 
{
                for (int j = 1; j < dim2 / 2; j++) {
                    Ae_sec_[0].add_inplace(Ae_sec_[j], parm->evaluator);
                }
                Ciphertext ct;
                parm->evaluator->rotate_rows(Ae_sec_[0].cipher_data[0], -dim1, alice_gk, ct);
                parm->evaluator->add_inplace(Ae_sec_[0].cipher_data[0], ct);
}
                C_sec[i] = Ae_sec_[0];
                delete[] Ae_sec_;
            }
            delete[] Ae_sec;
            vector<vector<uint64_t>> C(dim3, vector<uint64_t>(dim1));
#pragma omp parallel for
            for (int i = 0; i < dim3; i++) {
                random_ell_mat(C[i], 12);
                Plaintext pt; parm->encoder->encode(C[i], pt);
                parm->evaluator->sub_plain_inplace(C_sec[i].cipher_data[0], pt);
            }
            for (int i = 0; i < dim3; i++) {
                BFVLongCiphertext::send(io, C_sec + i);
            }
            delete[] C_sec;
            STOP_TIMER("FCHE")
        }
        std::cout << "comm: " << iopack->get_comm() - start << "\n";
        delete otpack;
        delete iopack;
        delete party;
        delete parm;
    } else { std::cout << "No party input\n"; }
}