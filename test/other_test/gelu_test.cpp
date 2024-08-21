#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include <seal/evaluator.h>
#include <utils.h>

using namespace sci;

BFVLongCiphertext f1(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x3, const BFVLongCiphertext &x4, Evaluator *evaluator) {
    BFVLongPlaintext parm0(
        parm, neg_mod(static_cast<int64_t>(-0.568686678 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm1(parm, neg_mod(static_cast<int64_t>(-0.529288810 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm2(parm, neg_mod(static_cast<int64_t>(-0.183509590 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm3(parm, neg_mod(static_cast<int64_t>(-0.028070202 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm4(parm, neg_mod(static_cast<int64_t>(-0.001597741 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE));
    BFVLongCiphertext ret = x4.multiply_plain(parm4, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret.add_inplace(ret2, evaluator);

    BFVLongCiphertext ret3 = x3.multiply_plain(parm3, evaluator);
    ret.add_inplace(ret3, evaluator);
    return ret;
}

BFVLongCiphertext f2(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x4, Evaluator *evaluator) {
    BFVLongPlaintext parm0(parm,
                           neg_mod(static_cast<int64_t>(0.001193207 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm1(parm, neg_mod(static_cast<int64_t>(0.5 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm2(parm, neg_mod(static_cast<int64_t>(0.385858026 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm4(parm, neg_mod(static_cast<int64_t>(-0.045101361 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE));
    BFVLongCiphertext ret = x4.multiply_plain(parm4, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret.add_inplace(ret2, evaluator);
    return ret;
}

BFVLongCiphertext f3(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x3, Evaluator *evaluator) {
    BFVLongPlaintext parm0(
        parm, neg_mod(static_cast<int64_t>(-0.438406187 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm1(parm, neg_mod(static_cast<int64_t>(1.340789252 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm2(parm, neg_mod(static_cast<int64_t>(-0.087184212 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        parm3(parm, neg_mod(static_cast<int64_t>(0.007334718 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE));
    BFVLongCiphertext ret = x3.multiply_plain(parm3, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret.add_inplace(ret2, evaluator);
    return ret;
}

void gelu(BFVKey *party, BFVLongCiphertext &ct_x, NetIO *io, FPMath *fpmath, Conversion *conv) {
    if (party->party == ALICE) {
        vector<uint64_t> f1_a = conv->he_to_ss_client(io, party);
        vector<uint64_t> f2_a = conv->he_to_ss_client(io, party);
        vector<uint64_t> f3_a = conv->he_to_ss_client(io, party);
        conv->Prime_to_Ring(party->party, f1_a.data(), f1_a.data(), f1_a.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        conv->Prime_to_Ring(party->party, f2_a.data(), f2_a.data(), f2_a.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        conv->Prime_to_Ring(party->party, f3_a.data(), f3_a.data(), f3_a.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
    } else {
        BFVLongCiphertext ct_x2 = ct_x.square(party->parm->evaluator); // with return ciphertext = 3
        ct_x.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x2.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x2.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext ct_x3 = ct_x2.multiply(ct_x, party->parm->evaluator); // with return ciphertext = 3
        ct_x3.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x3.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext ct_x4 = ct_x2.square(party->parm->evaluator);
        ct_x4.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x4.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext f1_res = f1(party->parm, ct_x, ct_x2, ct_x3, ct_x4, party->parm->evaluator);
        BFVLongCiphertext f2_res = f2(party->parm, ct_x, ct_x2, ct_x4, party->parm->evaluator);
        BFVLongCiphertext f3_res = f3(party->parm, ct_x, ct_x2, ct_x3, party->parm->evaluator);
        vector<uint64_t> f1_b = conv->he_to_ss_server(io, party->parm, f1_res);
        vector<uint64_t> f2_b = conv->he_to_ss_server(io, party->parm, f2_res);
        vector<uint64_t> f3_b = conv->he_to_ss_server(io, party->parm, f3_res);
        conv->Prime_to_Ring(party->party, f1_b.data(), f1_b.data(), f1_b.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        conv->Prime_to_Ring(party->party, f2_b.data(), f2_b.data(), f2_b.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        conv->Prime_to_Ring(party->party, f3_b.data(), f3_b.data(), f3_b.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
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

        int dim1 = 128, dim2 = 3072;
        vector<uint64_t> input_x(dim1 * dim2);
        random_modP_mat(input_x, default_prime_mod.at(31));
        BFVLongPlaintext pt_x(bfv_parm, input_x.data(), dim1 * dim2);
        BFVLongCiphertext ct_x(pt_x, party);
        gelu(party, ct_x, io, fpmath, conv);
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}