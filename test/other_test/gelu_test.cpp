#include "FixedPoint/bool-data.h"
#include "FixedPoint/fixed-math.h"
#include "FixedPoint/fixed-point.h"
#include "Utils/ezpc_scilib_tool.h"
#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include <cstdint>
#include <seal/evaluator.h>
#include <utils.h>
#define N_THREADS 24

using namespace sci;

INIT_TIMER

int64_t mod_inverse(int64_t a, int64_t m) {
    int64_t m0 = m, x0 = 0, x1 = 1;

    while (a > 1) {
        int64_t q = a / m;
        int64_t temp = m;
        m = a % m;
        a = temp;
        int64_t temp_x = x0;
        x0 = x1 - q * x0;
        x1 = temp_x;
    }

    return x1 < 0 ? x1 + m0 : x1;
}

BFVLongCiphertext f1(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x3, const BFVLongCiphertext &x4, Evaluator *evaluator) {
    BFVLongPlaintext parm0(parm,
                           neg_mod(static_cast<int64_t>(-0.568686678 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm1(parm, neg_mod(static_cast<int64_t>(-0.529288810 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm2(parm, neg_mod(static_cast<int64_t>(-0.183509590 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm3(parm, neg_mod(static_cast<int64_t>(-0.028070202 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm4(parm, neg_mod(static_cast<int64_t>(-0.001597741 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
    BFVLongCiphertext ret = x4.multiply_plain(parm4, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);

    BFVLongCiphertext ret3 = x3.multiply_plain(parm3, evaluator);
    ret.add_inplace(ret3, evaluator);
    return ret;
}

BFVLongCiphertext f2(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x4, Evaluator *evaluator) {
    BFVLongPlaintext parm0(parm, neg_mod(static_cast<int64_t>(0.001193207 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm1(parm, neg_mod(static_cast<int64_t>(0.5 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm2(parm, neg_mod(static_cast<int64_t>(0.385858026 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm4(parm, neg_mod(static_cast<int64_t>(-0.045101361 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
    BFVLongCiphertext ret = x4.multiply_plain(parm4, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);
    return ret;
}

BFVLongCiphertext f3(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2,
                     const BFVLongCiphertext &x3, Evaluator *evaluator) {
    BFVLongPlaintext parm0(parm,
                           neg_mod(static_cast<int64_t>(-0.438406187 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm1(parm, neg_mod(static_cast<int64_t>(1.340789252 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm2(parm, neg_mod(static_cast<int64_t>(-0.087184212 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        parm3(parm, neg_mod(static_cast<int64_t>(0.007334718 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
    BFVLongCiphertext ret = x3.multiply_plain(parm3, evaluator);
    ret.add_plain_inplace(parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);
    return ret;
}

void LT_thread(uint64_t *x, int x_party, uint64_t *y, int y_party, uint8_t *out, int num_ops, FPMath *fpmath) {
    FixArray fix_x = fpmath->fix->input(x_party, num_ops, x, true, DEFAULT_ELL, DEFAULT_SCALE);
    FixArray fix_y = fpmath->fix->input(y_party, num_ops, y, true, DEFAULT_ELL, DEFAULT_SCALE);
    BoolArray bool_out = fpmath->fix->LT(fix_x, fix_y);
    memcpy(out, bool_out.data, sizeof(uint8_t) * num_ops);
}

void LT(int party, uint64_t *x, int x_party, uint64_t *y, int y_party, BoolArray &output, int dim, FPMath **fpmath) {
    output = BoolArray(party, dim);
    std::thread threads[N_THREADS];
    int chunk_size = dim / N_THREADS;
    for (int i = 0; i < N_THREADS; i++) {
        int offset = i * chunk_size;
        int lnum_ops = (i == (N_THREADS - 1)) ? dim - offset : chunk_size;
        threads[i] =
            std::thread(LT_thread, x + offset, x_party, y + offset, y_party, output.data + offset, lnum_ops, fpmath[i]);
    }
    for (int i = 0; i < N_THREADS; ++i) {
        threads[i].join();
    }
}

BoolArray LT(int party, FixArray &x, uint64_t y, FixOp *fix, FPMath **fpmath) {
    FixArray fix_y = fix->input(PUBLIC, x.size, y, true, x.ell, x.s);
    BoolArray output;
    LT(party, x.data, x.party, fix_y.data, fix_y.party, output, x.size, fpmath);
    return output;
}

BoolArray LT(int party, uint64_t x, FixArray &y, FixOp *fix, FPMath **fpmath) {
    FixArray fix_x = fix->input(PUBLIC, y.size, x, true, y.ell, y.s);
    BoolArray output;
    LT(party, fix_x.data, fix_x.party, y.data, y.party, output, y.size, fpmath);
    return output;
}

void gelu(BFVKey *party, BFVLongCiphertext &ct_x, FPMath **fpmath, Conversion *conv, AuxProtocols *aux,
          BoolOp *boolop) {
    NetIO *io = fpmath[0]->iopack->io;
    if (party->party == ALICE) {
        vector<uint64_t> x = conv->he_to_ss_client(io, party);
        auto size_x = x.size();
        conv->Prime_to_Ring(party->party, N_THREADS, x.data(), x.data(), size_x, DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);

        FixArray fix_x = fpmath[0]->fix->input(party->party, size_x, x.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        // START_TIMER
        // BoolArray S0 = fpmath[0]->fix->LT(
        //               fix_x, neg_mod(static_cast<int64_t>(-5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        //           S1 = fpmath[0]->fix->LT(
        //               fix_x, neg_mod(static_cast<int64_t>(-sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL <<
        //               DEFAULT_SCALE)),
        //           S2 = fpmath[0]->fix->LT(
        //               fpmath[0]->fix->input(
        //                   PUBLIC, size_x,
        //                   neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
        //                   true, DEFAULT_ELL, DEFAULT_SCALE),
        //               fix_x),
        //           S3 = fpmath[0]->fix->LT(
        //               fpmath[0]->fix->input(
        //                   PUBLIC, size_x,
        //                   neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
        //                   true, DEFAULT_ELL, DEFAULT_SCALE),
        //               fix_x);
        BoolArray S0 = LT(party->party, fix_x,
                          neg_mod(static_cast<int64_t>(-5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fpmath[0]->fix, fpmath),
                  S1 = LT(party->party, fix_x,
                          neg_mod(static_cast<int64_t>(-sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fpmath[0]->fix, fpmath),
                  S2 = LT(party->party,
                          neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fix_x, fpmath[0]->fix, fpmath),
                  S3 = LT(party->party,
                          neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE), fix_x,
                          fpmath[0]->fix, fpmath);
        // STOP_TIMER("LT")
        BoolArray sign_b0 = S0, sign_b1 = boolop->XOR(S0, S1), sign_b2 = boolop->XOR(S1, S2),
                  sign_b3 = boolop->XOR(S2, S3), sign_b4 = S3;
        vector<uint64_t> b0(size_x), b1(size_x), b2(size_x), b3(size_x), b4(size_x);
        // START_TIMER
        aux->B2A(sign_b0.data, b0.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b1.data, b1.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b2.data, b2.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b3.data, b3.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b4.data, b4.data(), static_cast<int32_t>(size_x), 1);
        // STOP_TIMER("B2A")
        for (size_t i = 0; i < size_x; i++) {
            b0[i] = (b0[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b1[i] = (b1[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b2[i] = (b2[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b3[i] = (b3[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b4[i] = (b4[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
        }
        BFVLongPlaintext b0_plain(party->parm, b0), b1_plain(party->parm, b1), b2_plain(party->parm, b2),
            b3_plain(party->parm, b3), b4_plain(party->parm, b4);
        BFVLongCiphertext b0_sec_a(b0_plain, party), b1_sec_a(b1_plain, party), b2_sec_a(b2_plain, party),
            b3_sec_a(b3_plain, party), b4_sec_a(b4_plain, party);
        BFVLongCiphertext::send(io, &b0_sec_a);
        BFVLongCiphertext::send(io, &b1_sec_a);
        BFVLongCiphertext::send(io, &b2_sec_a);
        BFVLongCiphertext::send(io, &b3_sec_a);
        BFVLongCiphertext::send(io, &b4_sec_a);
    } else {
        uint64_t scale_inv = mod_inverse(1ULL << DEFAULT_SCALE, party->parm->plain_mod);
        BFVLongPlaintext scale_inv_plain(party->parm, scale_inv);
        BFVLongCiphertext ct_x2 = ct_x.square(party->parm->evaluator); // with return ciphertext = 3
        ct_x2.multiply_plain_inplace(scale_inv_plain, party->parm->evaluator);
        ct_x.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x2.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x2.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext ct_x3 = ct_x2.multiply(ct_x, party->parm->evaluator); // with return ciphertext = 3
        ct_x3.multiply_plain_inplace(scale_inv_plain, party->parm->evaluator);
        ct_x3.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x3.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext ct_x4 = ct_x2.square(party->parm->evaluator);
        ct_x4.multiply_plain_inplace(scale_inv_plain, party->parm->evaluator);
        ct_x4.mod_switch_to_next_inplace(party->parm->evaluator);
        ct_x4.relinearize_inplace(party->parm->evaluator, party->relin_keys);

        BFVLongCiphertext f1_res = f1(party->parm, ct_x, ct_x2, ct_x3, ct_x4, party->parm->evaluator);
        BFVLongCiphertext f2_res = f2(party->parm, ct_x, ct_x2, ct_x4, party->parm->evaluator);
        BFVLongCiphertext f3_res = f3(party->parm, ct_x, ct_x2, ct_x3, party->parm->evaluator);
        vector<uint64_t> x = conv->he_to_ss_server(io, party->parm, ct_x);
        auto size_x = x.size();
        conv->Prime_to_Ring(party->party, N_THREADS, x.data(), x.data(), size_x, DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        FixArray fix_x = fpmath[0]->fix->input(party->party, size_x, x.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        // START_TIMER
        // BoolArray S0 = fpmath[0]->fix->LT(
        //               fix_x, neg_mod(static_cast<int64_t>(-5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE)),
        //           S1 = fpmath[0]->fix->LT(
        //               fix_x, neg_mod(static_cast<int64_t>(-sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL <<
        //               DEFAULT_SCALE)),
        //           S2 = fpmath[0]->fix->LT(
        //               fpmath[0]->fix->input(
        //                   PUBLIC, size_x,
        //                   neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
        //                   true, DEFAULT_ELL, DEFAULT_SCALE),
        //               fix_x),
        //           S3 = fpmath[0]->fix->LT(
        //               fpmath[0]->fix->input(
        //                   PUBLIC, size_x,
        //                   neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
        //                   true, DEFAULT_ELL, DEFAULT_SCALE),
        //               fix_x);
        BoolArray S0 = LT(party->party, fix_x,
                          neg_mod(static_cast<int64_t>(-5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fpmath[0]->fix, fpmath),
                  S1 = LT(party->party, fix_x,
                          neg_mod(static_cast<int64_t>(-sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fpmath[0]->fix, fpmath),
                  S2 = LT(party->party,
                          neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          fix_x, fpmath[0]->fix, fpmath),
                  S3 = LT(party->party,
                          neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE), fix_x,
                          fpmath[0]->fix, fpmath);
        // STOP_TIMER("LT")
        BoolArray sign_b0 = S0, sign_b1 = boolop->XOR(S0, S1), sign_b2 = boolop->XOR(S1, S2),
                  sign_b3 = boolop->XOR(S2, S3), sign_b4 = S3;
        vector<uint64_t> b0(size_x), b1(size_x), b2(size_x), b3(size_x), b4(size_x);
        // START_TIMER
        aux->B2A(sign_b0.data, b0.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b1.data, b1.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b2.data, b2.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b3.data, b3.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b4.data, b4.data(), static_cast<int32_t>(size_x), 1);
        // STOP_TIMER("B2A")
        for (size_t i = 0; i < size_x; i++) {
            b0[i] = (b0[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b1[i] = (b1[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b2[i] = (b2[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b3[i] = (b3[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b4[i] = (b4[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
        }
        BFVLongCiphertext b0_sec_a, b1_sec_a, b2_sec_a, b3_sec_a, b4_sec_a;
        BFVLongCiphertext::recv(io, &b0_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b1_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b2_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b3_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b4_sec_a, party->parm->context);

        b0_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        b1_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        b2_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        b3_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        b4_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        BFVLongCiphertext gelu_sec_a = b4_sec_a.multiply(ct_x, party->parm->evaluator);
        gelu_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        gelu_sec_a.relinearize_inplace(party->parm->evaluator, party->relin_keys);
        b0_sec_a.mod_switch_to_next_inplace(party->parm->evaluator);
        gelu_sec_a.add_inplace(b0_sec_a, party->parm->evaluator);

        b1_sec_a.mod_switch_to_inplace(f1_res.parms_id(), party->parm->evaluator);
        b1_sec_a.multiply(f1_res, party->parm->evaluator);
        gelu_sec_a.add_inplace(b1_sec_a, party->parm->evaluator);

        b2_sec_a.mod_switch_to_inplace(f2_res.parms_id(), party->parm->evaluator);
        b2_sec_a.multiply(f2_res, party->parm->evaluator);
        gelu_sec_a.add_inplace(b2_sec_a, party->parm->evaluator);

        b3_sec_a.mod_switch_to_inplace(f3_res.parms_id(), party->parm->evaluator);
        b3_sec_a.multiply(f3_res, party->parm->evaluator);
        gelu_sec_a.add_inplace(b3_sec_a, party->parm->evaluator);
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
        IOPack *iopack[N_THREADS]; // = new IOPack(party_, 56789, ip);
        OTPack *otpack[N_THREADS]; // = new OTPack(iopack, party_);
        FPMath *fpmath[N_THREADS]; // = new FPMath(party_, iopack, otpack);
        for (int i = 0; i < N_THREADS; i++) {
            iopack[i] = new IOPack(party_, 56789 + i, ip);
            otpack[i] = new OTPack(iopack[i], party_);
            fpmath[i] = new FPMath(party_, iopack[i], otpack[i]);
        }

        Conversion *conv = new Conversion();
        AuxProtocols *aux = new AuxProtocols(party_, iopack[0], otpack[0]);
        BoolOp *boolop = new BoolOp(party_, iopack[0], otpack[0]);
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);

        int dim1 = 128, dim2 = 3072;
        vector<uint64_t> input_x(dim1 * dim2);
        random_modP_mat(input_x, default_prime_mod.at(31));
        BFVLongPlaintext pt_x(bfv_parm, input_x.data(), dim1 * dim2);
        BFVLongCiphertext ct_x(pt_x, party);
        size_t start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            start += iopack[i]->get_comm();
        }
        START_TIMER
        gelu(party, ct_x, fpmath, conv, aux, boolop);
        STOP_TIMER("gelu")
        size_t end = 0;
        for (int i = 0; i < N_THREADS; i++) {
            end += iopack[i]->get_comm();
        }
        std::cout << "comm: " << end - start << "\n";
        delete party;
        delete bfv_parm;
        delete boolop;
        delete aux;
        delete conv;
        for (int i = 0; i < N_THREADS; i++) {
            delete iopack[i];
            delete otpack[i];
            delete fpmath[i];
        }
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}