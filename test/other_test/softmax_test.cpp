#include "FixedPoint/fixed-math.h"
#include "FixedPoint/fixed-point.h"
#include "Utils/constants.h"
#include "protocols/fixed-protocol.h"
#include "utils/conversion.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <cstdint>
#include <utils.h>

using namespace sci;

// 扩展欧几里得算法，求解 ax + by = gcd(a, b)
uint64_t exgcd(uint64_t a, uint64_t b, uint64_t &x, uint64_t &y) {
    x = 1; y = 0;
    uint64_t x1 = 0, y1 = 1;
    while (b != 0) {
        uint64_t m = a / b;
        uint64_t t = b;
        b = a % b;
        a = t;
        uint64_t t1 = x;
        x = y1 - m * x;
        y = t1;
    }
    return a; // 返回最大公约数
}

// 求 a 模 p 的逆元，若不存在则返回 -1
uint64_t mod_inverse(uint64_t a, uint64_t p) {
    uint64_t x, y;
    uint64_t d = exgcd(a, p, x, y);
    if (d != 1) return -1;
    return (x % p + p) % p;
}

vector<uint64_t> softmax(BFVKey *party, vector<uint64_t> &input, vector<uint64_t> &output, int dim1, int dim2, NetIO *io,
             FPMath *fpmath, Conversion *conv) {
    int size = dim1 * dim2;
    assert(input.size() == size && output.size() == size);
    FixArray fix_inp = fpmath->fix->input(PUBLIC, size, input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
    FixArray exp_inp = fpmath->location_exp(fix_inp, DEFAULT_SCALE, DEFAULT_SCALE); // 4096, 4097, 4098, 4099
    if (party->party == ALICE) {
        BFVLongCiphertext exp_sec_b = conv->ss_to_he_server(party->parm, io, exp_inp.data, exp_inp.size, false);
        vector<uint64_t> R(size);
        // random_modP_mat(R, party->parm->plain_mod);
        BFVLongPlaintext R_plain(party->parm, R);
        auto exp_R_sec_b = exp_sec_b.add_plain(R_plain, party->parm->evaluator);
        vector<FixArray> fix_R(dim1);
#pragma omp parallel for
        for (int i = 0; i < dim1; i++) {
            fix_R[i] = fpmath->fix->input(party->party, dim2, R.data() + i * dim2, true, DEFAULT_ELL, DEFAULT_SCALE);
        }
        FixArray fix_SR = fpmath->fix->tree_sum(fix_R);
        BFVLongPlaintext SR_plain(party->parm, fix_SR.data, fix_SR.size);
        BFVLongCiphertext SR_sec_a(SR_plain, party);
        BFVLongCiphertext::send(io, &exp_R_sec_b);
        BFVLongCiphertext::send(io, &SR_sec_a);

        BFVLongCiphertext S_exp_V, V_sec_b;
        BFVLongCiphertext::recv(io, &S_exp_V, party->parm->context);
        BFVLongCiphertext::recv(io, &V_sec_b, party->parm->context);
        BFVLongPlaintext S_exp_V_plain = S_exp_V.decrypt(party);
        vector<uint64_t> Sexp_V = S_exp_V_plain.decode_uint(party->parm), Sexp_V_expand(dim1 * dim2);
#pragma omp parallel for
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                Sexp_V_expand[i * dim2 + j] = mod_inverse(Sexp_V[i], party->parm->plain_mod);
            }
        }
        BFVLongPlaintext Sexp_expand_plain(party->parm, Sexp_V_expand);
        V_sec_b.multiply_plain_inplace(Sexp_expand_plain, party->parm->evaluator);
        exp_sec_b.multiply_inplace(V_sec_b, party->parm->evaluator);

        BFVLongCiphertext::send(io, &exp_sec_b);
        vector<uint64_t> ret = conv->he_to_ss_server(io, party->parm, exp_sec_b);
        std::cout << ret[0] << " " << ret[1] << "\n" << ret[2] << " " << ret[3] << "\n";
        return ret;
    } else {
        conv->ss_to_he_client(party, io, exp_inp.data, exp_inp.size, DEFAULT_ELL);

        BFVLongCiphertext exp_sec_b, R_sec_a, SR_sec_a;
        BFVLongCiphertext::recv(io, &exp_sec_b, party->parm->context);
        BFVLongCiphertext::recv(io, &SR_sec_a, party->parm->context);
        BFVLongPlaintext exp_R_plain = exp_sec_b.decrypt(party);
        vector<uint64_t> exp_R = exp_R_plain.decode_uint(party->parm);
        vector<FixArray> fix_exp_R(dim1);
#pragma omp parallel for
        for (int i = 0; i < dim1; i++) {
            fix_exp_R[i] =
                fpmath->fix->input(party->party, dim2, exp_R.data() + i * dim2, true, DEFAULT_ELL, DEFAULT_SCALE);
        }
        FixArray fix_S_exp_R = fpmath->fix->tree_sum(fix_exp_R);
        BFVLongPlaintext S_exp_R_plain(party->parm, fix_S_exp_R.data, fix_S_exp_R.size);
        SR_sec_a.negate_inplace(party->parm->evaluator);
        SR_sec_a.add_plain_inplace(S_exp_R_plain, party->parm->evaluator);
        vector<uint64_t> V(dim1, 1), V_expand(dim1 * dim2);
        // random_modP_mat(V, party->parm->plain_mod);
#pragma omp parallel for
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                V_expand[i * dim2 + j] = V[i];
            }
        }
        BFVLongPlaintext V_plain(party->parm, V), V_expand_plain(party->parm, V_expand);
        SR_sec_a.sub_plain_inplace(V_plain, party->parm->evaluator);
        BFVLongCiphertext V_sec_b(V_expand_plain, party);
        BFVLongCiphertext::send(io, &SR_sec_a);
        BFVLongCiphertext::send(io, &V_sec_b);


        BFVLongCiphertext exp_sec_b_;
        BFVLongCiphertext::recv(io, &exp_sec_b_, party->parm->context);
        BFVLongPlaintext exp_plain = exp_sec_b_.decrypt(party);
        auto exp_ = exp_plain.decode_uint(party->parm);
        std::cout << exp_[0] << " " << exp_[1] << "\n" << exp_[2] << " " << exp_[3] << "\n";
        vector<uint64_t> ret = conv->he_to_ss_client(io, party);
        std::cout << ret[0] << " " << ret[1] << "\n" << ret[2] << " " << ret[3] << "\n";
        return ret;
    }
}

int main(int argc, const char **argv) {
    if (argc > 1) {
        int party_ = argv[1][0] - '0';
        assert(party_ == ALICE || party_ == BOB);
        party_ == ALICE ? std::cout << "Party: ALICE\n" : std::cout << "Party: BOB\n";
        string ip = "127.0.0.1";
        if (argc > 2) {
            ip = argv[2];
        }
        IOPack *iopack = new IOPack(party_, 56789, ip);
        OTPack *otpack = new OTPack(iopack, party_);
        NetIO *io = iopack->io;
        FPMath *fpmath = new FPMath(party_, iopack, otpack);
        Conversion *conv = new Conversion();
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);

        int dim1 = 2, dim2 = 2;
        vector<uint64_t> input = {1, 2, 3, 4}, output(dim1 * dim2);
        // random_ell_mat(input, DEFAULT_ELL);
        auto start = iopack->get_comm();
        INIT_TIMER
        START_TIMER
        softmax(party, input, output, dim1, dim2, io, fpmath, conv);
        STOP_TIMER("softmax")
        std::cout << "comm: " << iopack->get_comm() - start << "\n";

        delete party;
        delete bfv_parm; 
        delete conv;
        delete fpmath;
        io = nullptr;
        delete otpack;
        delete iopack;
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}