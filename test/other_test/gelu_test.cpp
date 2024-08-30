#include "protocols/fixed-protocol.h"
#include <utils.h>
#define N_THREADS 12

using namespace sci;

INIT_TIMER

BFVLongPlaintext f1_parm0, f1_parm1, f1_parm2, f1_parm3, f1_parm4, f2_parm0, f2_parm1, f2_parm2, f2_parm4, f3_parm0,
    f3_parm1, f3_parm2, f3_parm3;

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

void LT_thread(uint64_t *x, int x_party, uint64_t *y, int y_party, uint8_t *out, int num_ops, FPMath *fpmath) {
    FixArray fix_x = fpmath->fix->input(x_party, num_ops, x, true, DEFAULT_ELL, DEFAULT_SCALE);
    FixArray fix_y = fpmath->fix->input(y_party, num_ops, y, true, DEFAULT_ELL, DEFAULT_SCALE);
    BoolArray bool_out = fpmath->fix->LT(fix_x, fix_y);
    memcpy(out, bool_out.data, sizeof(uint8_t) * num_ops);
}

BoolArray LT(int party, FixArray &x, FixArray &y, int dim, FPMath **fpmath) {
    BoolArray output(party, dim);
    std::thread threads[N_THREADS];
    int chunk_size = dim / N_THREADS;
    for (int i = 0; i < N_THREADS; i++) {
        int offset = i * chunk_size;
        int lnum_ops = (i == (N_THREADS - 1)) ? dim - offset : chunk_size;
        threads[i] = std::thread(LT_thread, x.data + offset, x.party, y.data + offset, y.party, output.data + offset,
                                 lnum_ops, fpmath[i]);
    }
    for (int i = 0; i < N_THREADS; ++i) {
        threads[i].join();
    }
    return output;
}

BoolArray LT(int party, FixArray &x, uint64_t y, FPMath **fpmath) {
    FixArray fix_y = fpmath[0]->fix->input(PUBLIC, x.size, y, true, x.ell, x.s);
    return LT(party, x, fix_y, x.size, fpmath);
}

BoolArray LT(int party, uint64_t x, FixArray &y, FPMath **fpmath) {
    FixArray fix_x = fpmath[0]->fix->input(PUBLIC, y.size, x, true, y.ell, y.s);
    return LT(party, fix_x, y, y.size, fpmath);
}

void AND_thread(uint8_t *x, int x_party, uint8_t *y, int y_party, uint8_t *out, int num_ops, FPMath *fpmath) {
    BoolArray bool_x = fpmath->bool_op->input(x_party, num_ops, x);
    BoolArray bool_y = fpmath->bool_op->input(y_party, num_ops, y);
    BoolArray bool_out = fpmath->bool_op->AND(bool_x, bool_y);
    memcpy(out, bool_out.data, sizeof(uint8_t) * num_ops);
}

BoolArray AND(int party, BoolArray &x, BoolArray &y, FPMath **fpmath) {
    BoolArray output(party, x.size);
    std::thread threads[N_THREADS];
    int chunk_size = output.size / N_THREADS;
    for (int i = 0; i < N_THREADS; i++) {
        int offset = i * chunk_size;
        int lnum_ops = (i == (N_THREADS - 1)) ? output.size - offset : chunk_size;
        threads[i] = std::thread(AND_thread, x.data + offset, x.party, y.data + offset, y.party, output.data + offset,
                                 lnum_ops, fpmath[i]);
    }
    for (int i = 0; i < N_THREADS; ++i) {
        threads[i].join();
    }
    return output;
}

void f1(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2, const BFVLongCiphertext &x3,
        const BFVLongCiphertext &x4, BFVLongCiphertext &ret, Evaluator *evaluator) {
    ret = x4.multiply_plain(f1_parm4, evaluator);
    ret.add_plain_inplace(f1_parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(f1_parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(f1_parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);

    BFVLongCiphertext ret3 = x3.multiply_plain(f1_parm3, evaluator);
    ret.add_inplace(ret3, evaluator);
}

void f2(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2, const BFVLongCiphertext &x4,
        BFVLongCiphertext &ret, Evaluator *evaluator) {
    ret = x4.multiply_plain(f2_parm4, evaluator);
    ret.add_plain_inplace(f2_parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(f2_parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(f2_parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);
}

void f3(BFVParm *parm, const BFVLongCiphertext &x, const BFVLongCiphertext &x2, const BFVLongCiphertext &x3,
        BFVLongCiphertext &ret, Evaluator *evaluator) {
    ret = x3.multiply_plain(f3_parm3, evaluator);
    ret.add_plain_inplace(f3_parm0, evaluator);

    BFVLongCiphertext ret1 = x.multiply_plain(f3_parm1, evaluator);
    ret1.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret1, evaluator);

    BFVLongCiphertext ret2 = x2.multiply_plain(f3_parm2, evaluator);
    ret2.mod_switch_to_next_inplace(evaluator);
    ret.add_inplace(ret2, evaluator);
}

void gelu(BFVKey *party, BFVLongCiphertext &ct_x, FPMath **fpmath, Conversion *conv, AuxProtocols *aux,
          BoolOp *boolop) {
    NetIO *io = fpmath[0]->iopack->io;
    if (party->party == ALICE) {
        auto start_conv = std::chrono::high_resolution_clock::now();
        vector<uint64_t> x = conv->he_to_ss_client(io, party);
        auto size_x = x.size();
        conv->Prime_to_Ring(party->party, N_THREADS, x.data(), x.data(), size_x, DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        auto stop_conv = std::chrono::high_resolution_clock::now() - start_conv;

        auto start_comp = std::chrono::high_resolution_clock::now();
        BoolArray all_zero = fpmath[0]->bool_op->input(PUBLIC, size_x, uint8_t(0));
        fpmath[0]->fix->input(PUBLIC, size_x, static_cast<uint64_t>(0), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_x = fpmath[0]->fix->input(party->party, size_x, x.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        auto msb_x = LT(party->party, fix_x, 0, fpmath);
        FixArray neg_x = fpmath[0]->fix->mul(fix_x, -1);
        auto abs_x = fpmath[0]->fix->if_else(msb_x, neg_x, fix_x);
        BoolArray gt_zero = AND(party->party, msb_x, all_zero, fpmath),
                  S2 = LT(party->party,
                          neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          abs_x, fpmath),
                  S3 = LT(party->party,
                          neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE), abs_x,
                          fpmath);
        BoolArray S0 = AND(party->party, gt_zero, S3, fpmath), S1 = AND(party->party, gt_zero, S2, fpmath);
        BoolArray sign_b0 = S0, sign_b1 = boolop->XOR(S0, S1), sign_b2 = boolop->XOR(S1, S2),
                  sign_b3 = boolop->XOR(S2, S3), sign_b4 = S3;
        vector<uint64_t> b0(size_x), b1(size_x), b2(size_x), b3(size_x), b4(size_x);
        aux->B2A(sign_b0.data, b0.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b1.data, b1.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b2.data, b2.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b3.data, b3.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b4.data, b4.data(), static_cast<int32_t>(size_x), 1);
#pragma omp parallel for
        for (size_t i = 0; i < size_x; i++) {
            b0[i] = (b0[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b1[i] = (b1[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b2[i] = (b2[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b3[i] = (b3[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b4[i] = (b4[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
        }
        auto stop_comp = std::chrono::high_resolution_clock::now() - start_comp;

        auto start_enc = std::chrono::high_resolution_clock::now();
        BFVLongPlaintext b0_plain(party->parm, b0), b1_plain(party->parm, b1), b2_plain(party->parm, b2),
            b3_plain(party->parm, b3), b4_plain(party->parm, b4);
        BFVLongCiphertext b0_sec_a(b0_plain, party), b1_sec_a(b1_plain, party), b2_sec_a(b2_plain, party),
            b3_sec_a(b3_plain, party), b4_sec_a(b4_plain, party);
        auto stop_enc = std::chrono::high_resolution_clock::now() - start_enc;

        auto start_send = std::chrono::high_resolution_clock::now();
        BFVLongCiphertext::send(io, &b0_sec_a);
        BFVLongCiphertext::send(io, &b1_sec_a);
        BFVLongCiphertext::send(io, &b2_sec_a);
        BFVLongCiphertext::send(io, &b3_sec_a);
        BFVLongCiphertext::send(io, &b4_sec_a);
        auto stop_send = std::chrono::high_resolution_clock::now() - start_send;
        auto time = stop_comp + stop_enc + stop_send;

        std::cout << "conv cost: " << stop_conv.count() / 1000000 << " ms\n";
        // std::cout << "enc cost: " << stop_enc.count() / 1000000 << " ms\n";
        std::cout << "time cost: " << time.count() / 1000000 << " ms\n";
    } else {
        uint64_t scale_inv = mod_inverse(1ULL << DEFAULT_SCALE, party->parm->plain_mod);
        BFVLongPlaintext scale_inv_plain(party->parm, scale_inv);

        auto start_cc = std::chrono::high_resolution_clock::now();
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
        auto stop_cc = std::chrono::high_resolution_clock::now() - start_cc;

        vector<std::thread> f_threads(3);
        BFVLongCiphertext f1_res, f2_res, f3_res;
        auto start_f = std::chrono::high_resolution_clock::now();
        f_threads[0] = std::thread(f1, party->parm, std::ref(ct_x), std::ref(ct_x2), std::ref(ct_x3), std::ref(ct_x4),
                                   std::ref(f1_res), party->parm->evaluator);
        f_threads[1] = std::thread(f2, party->parm, std::ref(ct_x), std::ref(ct_x2), std::ref(ct_x4), std::ref(f2_res),
                                   party->parm->evaluator);
        f_threads[2] = std::thread(f3, party->parm, std::ref(ct_x), std::ref(ct_x2), std::ref(ct_x3), std::ref(f3_res),
                                   party->parm->evaluator);
        f_threads[0].join();
        f_threads[1].join();
        f_threads[2].join();
        auto stop_f = std::chrono::high_resolution_clock::now() - start_f;

        auto start_conv = std::chrono::high_resolution_clock::now();
        vector<uint64_t> x = conv->he_to_ss_server(io, party->parm, ct_x);
        auto size_x = x.size();
        conv->Prime_to_Ring(party->party, N_THREADS, x.data(), x.data(), size_x, DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        auto stop_conv = std::chrono::high_resolution_clock::now() - start_conv;

        auto start_cmp = std::chrono::high_resolution_clock::now();
        BoolArray all_zero = fpmath[0]->bool_op->input(PUBLIC, size_x, uint8_t(0));
        fpmath[0]->fix->input(PUBLIC, size_x, static_cast<uint64_t>(0), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_x = fpmath[0]->fix->input(party->party, size_x, x.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        auto msb_x = LT(party->party, fix_x, 0, fpmath);
        FixArray neg_x = fpmath[0]->fix->mul(fix_x, -1);
        auto abs_x = fpmath[0]->fix->if_else(msb_x, neg_x, fix_x);
        BoolArray gt_zero = AND(party->party, msb_x, all_zero, fpmath),
                  S2 = LT(party->party,
                          neg_mod(static_cast<int64_t>(sqrt(2.) * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE),
                          abs_x, fpmath),
                  S3 = LT(party->party,
                          neg_mod(static_cast<int64_t>(5.075 * (1ULL << DEFAULT_SCALE)), 1ULL << DEFAULT_SCALE), abs_x,
                          fpmath);
        BoolArray S0 = AND(party->party, gt_zero, S3, fpmath), S1 = AND(party->party, gt_zero, S2, fpmath);
        BoolArray sign_b0 = S0, sign_b1 = boolop->XOR(S0, S1), sign_b2 = boolop->XOR(S1, S2),
                  sign_b3 = boolop->XOR(S2, S3), sign_b4 = S3;
        vector<uint64_t> b0(size_x), b1(size_x), b2(size_x), b3(size_x), b4(size_x);
        aux->B2A(sign_b0.data, b0.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b1.data, b1.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b2.data, b2.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b3.data, b3.data(), static_cast<int32_t>(size_x), 1);
        aux->B2A(sign_b4.data, b4.data(), static_cast<int32_t>(size_x), 1);
#pragma omp parallel for
        for (size_t i = 0; i < size_x; i++) {
            b0[i] = (b0[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b1[i] = (b1[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b2[i] = (b2[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b3[i] = (b3[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
            b4[i] = (b4[i] - (1ULL << DEFAULT_SCALE)) % party->parm->plain_mod;
        }
        auto stop_cmp = std::chrono::high_resolution_clock::now() - start_cmp;

        BFVLongCiphertext b0_sec_a, b1_sec_a, b2_sec_a, b3_sec_a, b4_sec_a;
        BFVLongCiphertext::recv(io, &b0_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b1_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b2_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b3_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &b4_sec_a, party->parm->context);

        auto start_fgelu = std::chrono::high_resolution_clock::now();
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
        auto stop_fgelu = std::chrono::high_resolution_clock::now() - start_fgelu;
        auto time = stop_cc + stop_f + stop_cmp + stop_fgelu;
        // std::cout << "cipher-cipher: " << stop_cc.count() / 1000000 << " ms\n"
        //           << "f: " << stop_f.count() / 1000000 << " ms\n"
        //           << "conversion: " << stop_conv.count() / 1000000 << " ms\n"
        //           << "compare: " << stop_cmp.count() / 1000000 << " ms\n"
        //           << "gelu: " << stop_fgelu.count() / 1000000 << " ms\n";
        std::cout << "conv cost: " << stop_conv.count() / 1000000 << " ms\n";
        std::cout << "time cost: " << time.count() / 1000000 << " ms\n";
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
        BFVParm *parm =
            new BFVParm(8192, {40, 30, 30, 40},
                        default_prime_mod.at(29)); // new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, parm);

        f1_parm0 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.568686678 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f1_parm1 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.529288810 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f1_parm2 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.183509590 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f1_parm3 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.028070202 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f1_parm4 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.001597741 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f2_parm0 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(0.001193207 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f2_parm1 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(0.500000000 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f2_parm2 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(0.385858026 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f2_parm4 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.045101361 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));
        f3_parm0 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.438406187 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        f3_parm1 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(1.340789252 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        f3_parm2 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(-0.087184212 * (1ULL << DEFAULT_SCALE)), parm->plain_mod)),
        f3_parm3 = BFVLongPlaintext(
            parm, neg_mod(static_cast<int64_t>(0.007334718 * (1ULL << DEFAULT_SCALE)), parm->plain_mod));

        int dim1 = 128, dim2 = 3072;
        vector<uint64_t> input_x(dim1 * dim2);
        random_modP_mat(input_x, default_prime_mod.at(31));
        BFVLongPlaintext pt_x(parm, input_x.data(), dim1 * dim2);
        BFVLongCiphertext ct_x(pt_x, party);
        size_t start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            start += iopack[i]->get_comm();
        }
        gelu(party, ct_x, fpmath, conv, aux, boolop);
        size_t end = 0;
        for (int i = 0; i < N_THREADS; i++) {
            end += iopack[i]->get_comm();
        }
        std::cout << "comm: " << end - start << "\n";
        delete party;
        delete parm;
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