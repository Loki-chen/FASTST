#include "FixedPoint/fixed-point.h"
#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <utils.h>

using namespace sci;

vector<uint64_t> layernorm(BFVKey *party, const vector<uint64_t> &input, const vector<uint64_t> &gamma,
                           const vector<uint64_t> &beta, int dim1, int dim2, NetIO *io, FPMath *fpmath,
                           Conversion *conv) {
    int size = dim1 * dim2;
    assert(input.size() == size);
    assert(gamma.size() == size);
    assert(beta.size() == size);
    vector<FixArray> fix_inp(dim1);
    for (int i = 0; i < dim1; i++) {
        fix_inp[i] = fpmath->fix->input(PUBLIC, dim2, input.data() + i * dim2, true, DEFAULT_ELL, DEFAULT_SCALE);
    }
    FixArray sigma = fpmath->fix->tree_sum(fix_inp),
             sigma_expand = FixArray(PUBLIC, size, true, DEFAULT_ELL, DEFAULT_SCALE);
    FixArray fix_A = fpmath->fix->input(PUBLIC, size, input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            fix_A.data[i * dim2 + j] *= dim2;
            sigma_expand.data[i * dim2 + j] = sigma.data[i];
        }
    }
    fix_A = fpmath->fix->sub(fix_A, sigma_expand);
    if (party->party == ALICE) {
        BFVLongCiphertext A_sec_b = conv->ss_to_he_server(party->parm, io, fix_A.data, fix_A.size, DEFAULT_ELL);
        BFVLongCiphertext A_s_sec_b = A_sec_b.square(party->parm->evaluator);

        vector<uint64_t> R(size);
        random_modP_mat(R, party->parm->plain_mod);
        BFVLongPlaintext R_plain(party->parm, R);
        A_s_sec_b.add_plain_inplace(R_plain, party->parm->evaluator);
        vector<FixArray> fix_R(dim1);
        for (int i = 0; i < dim1; i++) {
            fix_R[i] = fpmath->fix->input(party->party, dim2, R.data() + i * dim2, true, DEFAULT_ELL, DEFAULT_SCALE);
        }
        FixArray fix_SR = fpmath->fix->tree_sum(fix_R);
        BFVLongPlaintext SR_plain(party->parm, fix_SR.data, fix_SR.size);
        BFVLongCiphertext SR_sec_a(SR_plain, party);
        BFVLongCiphertext::send(io, &A_s_sec_b);
        BFVLongCiphertext::send(io, &SR_sec_a);
        

        vector<uint64_t> Ka = conv->he_to_ss_client(io, party);
        conv->Prime_to_Ring(party->party, Ka.data(), Ka.data(), Ka.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2,
                            DEFAULT_SCALE, fpmath);//return Ka;
        FixArray fix_Ka = fpmath->fix->input(party->party, Ka.size(), Ka.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray isqrt_Ka = fpmath->sqrt_(fix_Ka, true);
        FixArray isqrt_Ka_flat(party->party, size, true, DEFAULT_ELL, DEFAULT_SCALE);
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                isqrt_Ka_flat.data[i * dim2 + j] = isqrt_Ka.data[i];
            }
        }
        BFVLongCiphertext inv_K =
            conv->ss_to_he_server(party->parm, io, isqrt_Ka_flat.data, isqrt_Ka_flat.size, DEFAULT_ELL);
        A_sec_b.multiply_inplace(inv_K, party->parm->evaluator);
        vector<uint64_t> S(size);
        random_modP_mat(S, party->parm->plain_mod);
        BFVLongPlaintext S_plain(party->parm, S);
        BFVLongCiphertext S_sec_a(S_plain, party);
        A_sec_b.sub_plain_inplace(S_plain, party->parm->evaluator);
        BFVLongCiphertext::send(io, &A_sec_b);
        BFVLongCiphertext::send(io, &S_sec_a);

        vector<uint64_t> ret = conv->he_to_ss_client(io, party);
        conv->Prime_to_Ring(party->party, ret.data(), ret.data(), ret.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2,
                            DEFAULT_SCALE, fpmath);  // it's costly
        return ret;
    } else {
        conv->ss_to_he_client(party, io, fix_A.data, fix_A.size, DEFAULT_ELL);

        BFVLongCiphertext A_s_sec_b, SR_sec_a;
        BFVLongCiphertext::recv(io, &A_s_sec_b, party->parm->context);
        BFVLongCiphertext::recv(io, &SR_sec_a, party->parm->context);
        BFVLongPlaintext A_s_plain = A_s_sec_b.decrypt(party);
        vector<uint64_t> AR = A_s_plain.decode_uint(party->parm);
        vector<FixArray> fix_AR(dim1);
        for (int i = 0; i < dim1; i++) {
            fix_AR[i] = fpmath->fix->input(party->party, dim2, AR.data() + i * dim2, true, DEFAULT_ELL, DEFAULT_SCALE);
        }
        FixArray fix_SAR = fpmath->fix->tree_sum(fix_AR);
        BFVLongPlaintext SAR_plain(party->parm, fix_SAR.data, fix_SAR.size);
        SR_sec_a.negate_inplace(party->parm->evaluator);
        SR_sec_a.add_plain_inplace(SAR_plain, party->parm->evaluator);

        vector<uint64_t> Kb = conv->he_to_ss_server(io, party->parm, SR_sec_a);
        conv->Prime_to_Ring(party->party, Kb.data(), Kb.data(), Kb.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2,
                            DEFAULT_SCALE, fpmath);//return Kb;
        FixArray fix_Kb = fpmath->fix->input(party->party, Kb.size(), Kb.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray isqrt_Kb = fpmath->sqrt_(fix_Kb, true);
        FixArray isqrt_Kb_flat(party->party, size, true, DEFAULT_ELL, DEFAULT_SCALE);
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                isqrt_Kb_flat.data[i * dim2 + j] = isqrt_Kb.data[i];
            }
        }
        conv->ss_to_he_client(party, io, isqrt_Kb_flat.data, isqrt_Kb_flat.size, DEFAULT_ELL);

        BFVLongCiphertext A_sec_b, S_sec_a;
        BFVLongCiphertext::recv(io, &A_sec_b, party->parm->context);
        BFVLongCiphertext::recv(io, &S_sec_a, party->parm->context);
        BFVLongPlaintext A_plain = A_sec_b.decrypt(party), gamma_plain(party->parm, gamma),
                         beta_plain(party->parm, beta);
        S_sec_a.add_plain_inplace(A_plain, party->parm->evaluator);
        S_sec_a.multiply_plain_inplace(gamma_plain, party->parm->evaluator);
        S_sec_a.add_plain_inplace(beta_plain, party->parm->evaluator);
        vector<uint64_t> ret = conv->he_to_ss_server(io, party->parm, S_sec_a);
        conv->Prime_to_Ring(party->party, ret.data(), ret.data(), ret.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2,
                            DEFAULT_SCALE, fpmath);  // it's costly
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

        int dim1 = 128, dim2 = 768;
        vector<uint64_t> input(dim1 * dim2), gamma(dim1 * dim2), beta(dim1 * dim2);
        random_ell_mat(input, DEFAULT_ELL);
        random_ell_mat(gamma, DEFAULT_ELL);
        random_ell_mat(beta, DEFAULT_ELL);
        auto start = iopack->get_comm();
        INIT_TIMER
        START_TIMER
        auto output = layernorm(party, input, gamma, beta, dim1, dim2, io, fpmath, conv);
        STOP_TIMER("layernorm")
        std::cout << "comm: " << iopack->get_comm() - start << "\n";
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}