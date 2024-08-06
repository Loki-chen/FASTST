#include "fixed-ffn.h"
#include "FixedPoint/fixed-point.h"
#include "Utils/constants.h"
#include "Utils/ezpc_scilib_tool.h"
#include "Utils/prg.h"
#include "model.h"
#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <cstdint>
#include <stdexcept>

bfv_matrix random_sgn(size_t size, uint64_t ell_)
{
    bfv_matrix ret(size);
    matrix mask(size);
    random_mat(mask);
    for (size_t i = 0; i < size; i++)
    {
        ret[i] = mask[i] > 0 ? 1 : ell_ - 1;
    }
    return ret;
}

BFVLongCiphertext FixedFFN::f3(const BFVLongCiphertext &x1, const BFVLongCiphertext &x2,
                               const BFVLongCiphertext &x3) const
{
    uint64_t prime_parm0 = sci::neg_mod(static_cast<int64_t>(-0.438406187 * parm->plain_mod), parm->plain_mod),
             prime_parm1 = sci::neg_mod(static_cast<int64_t>(1.340789252 * parm->plain_mod), parm->plain_mod),
             prime_parm2 = sci::neg_mod(static_cast<int64_t>(-0.087184212 * parm->plain_mod), parm->plain_mod),
             prime_parm3 = sci::neg_mod(static_cast<int64_t>(0.007334718 * parm->plain_mod), parm->plain_mod);
    BFVLongPlaintext parm0(party->parm, prime_parm0), parm1(party->parm, prime_parm1), parm2(party->parm, prime_parm2),
        parm3(party->parm, prime_parm3);
    BFVLongCiphertext f3_ = x3.multiply_plain(parm3, party->parm->evaluator),
                      tmp1 = x2.multiply_plain(parm2, party->parm->evaluator),
                      tmp2 = x1.multiply_plain(parm1, party->parm->evaluator);
    f3_.add_inplace(tmp1, party->parm->evaluator);
    f3_.add_inplace(tmp2, party->parm->evaluator);
    f3_.add_plain_inplace(parm0, party->parm->evaluator);
    return f3_;
}

BFVLongCiphertext FixedFFN::f2(const BFVLongCiphertext &x1, const BFVLongCiphertext &x2,
                               const BFVLongCiphertext &x4) const
{
    uint64_t prime_parm0 = sci::neg_mod(static_cast<int64_t>(0.001193207 * parm->plain_mod), parm->plain_mod),
             prime_parm1 = sci::neg_mod(static_cast<int64_t>(0.5 * parm->plain_mod), parm->plain_mod),
             prime_parm2 = sci::neg_mod(static_cast<int64_t>(0.385858026 * parm->plain_mod), parm->plain_mod),
             prime_parm4 = sci::neg_mod(static_cast<int64_t>(-0.045101361 * parm->plain_mod), parm->plain_mod);
    BFVLongPlaintext parm0(party->parm, prime_parm0), parm1(party->parm, prime_parm1), parm2(party->parm, prime_parm2),
        parm4(party->parm, prime_parm4);
    BFVLongCiphertext f2_ = x4.multiply_plain(parm4, party->parm->evaluator),
                      tmp2 = x2.multiply_plain(parm2, party->parm->evaluator),
                      tmp3 = x1.multiply_plain(parm1, party->parm->evaluator);
    f2_.add_inplace(tmp2, party->parm->evaluator);
    f2_.add_inplace(tmp3, party->parm->evaluator);
    f2_.add_plain_inplace(parm0, party->parm->evaluator);
    return f2_;
}

BFVLongCiphertext FixedFFN::f1(const BFVLongCiphertext &x1, const BFVLongCiphertext &x2, const BFVLongCiphertext &x3,
                               const BFVLongCiphertext &x4) const
{
    uint64_t prime_parm0 = sci::neg_mod(static_cast<int64_t>(-0.568686678 * parm->plain_mod), parm->plain_mod),
             prime_parm1 = sci::neg_mod(static_cast<int64_t>(-0.529288810 * parm->plain_mod), parm->plain_mod),
             prime_parm2 = sci::neg_mod(static_cast<int64_t>(-0.183509590 * parm->plain_mod), parm->plain_mod),
             prime_parm3 = sci::neg_mod(static_cast<int64_t>(-0.028070202 * parm->plain_mod), parm->plain_mod),
             prime_parm4 = sci::neg_mod(static_cast<int64_t>(-0.001597741 * parm->plain_mod), parm->plain_mod);
    BFVLongPlaintext parm0(party->parm, prime_parm0), parm1(party->parm, prime_parm1), parm2(party->parm, prime_parm2),
        parm3(party->parm, prime_parm3), parm4(party->parm, prime_parm4);
    BFVLongCiphertext f1_ = x4.multiply_plain(parm4, party->parm->evaluator),
                      tmp1 = x3.multiply_plain(parm3, party->parm->evaluator),
                      tmp2 = x2.multiply_plain(parm2, party->parm->evaluator),
                      tmp3 = x1.multiply_plain(parm1, party->parm->evaluator);
    f1_.add_inplace(tmp1, party->parm->evaluator);
    f1_.add_inplace(tmp2, party->parm->evaluator);
    f1_.add_inplace(tmp3, party->parm->evaluator);
    f1_.add_plain_inplace(parm0, party->parm->evaluator);
    return f1_;
}

FixedFFN::FixedFFN(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                   Conversion *conv)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv)
{
    string W1_file = replace("bert.encoder.layer.LAYER.intermediate.dense.weight.txt", "LAYER", layer_str),
           W2_file = replace("bert.encoder.layer.LAYER.output.dense.weight.txt", "LAYER", layer_str),
           b1_file = replace("bert.encoder.layer.LAYER.intermediate.dense.bias.txt", "LAYER", layer_str),
           b2_file = replace("bert.encoder.layer.LAYER.output.dense.bias.txt", "LAYER", layer_str);
    try
    {
        bfv_matrix tmp_b1, tmp_b2;
        load_bfv_mat(W1, dir_path + W1_file);
        load_bfv_mat(W2, dir_path + W2_file);
        load_bfv_mat(tmp_b1, dir_path + b1_file);
        load_bfv_mat(tmp_b2, dir_path + b2_file);
        b1 = bfv_matrix(batch_size * ffn_dim);
        b2 = bfv_matrix(batch_size * d_module);
        for (size_t i = 0; i < batch_size; i++)
        {
            for (size_t j = 0; j < ffn_dim; j++)
            {
                b1[i * ffn_dim + j] = tmp_b1[j];
            }
            for (size_t j = 0; j < d_module; j++)
            {
                b2[i * d_module + j] = tmp_b2[j];
            }
        }
    }
    catch (std::runtime_error e)
    {
        std::cout << "[FFN] WARNINE: cannot open data file, generate data randonly\n";
        W1 = bfv_matrix(d_module * ffn_dim);
        W2 = bfv_matrix(ffn_dim * d_module);
        b1 = bfv_matrix(batch_size * ffn_dim);
        b2 = bfv_matrix(batch_size * d_module);
        random_ell_mat(W1, DEFAULT_ELL);
        random_ell_mat(W2, DEFAULT_ELL);
        random_ell_mat(b1, DEFAULT_ELL);
        random_ell_mat(b2, DEFAULT_ELL);
    }
}

BFVLongCiphertext FixedFFN::forward(const BFVLongCiphertext &input) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);
    size_t total_comm = io->counter;
    if (party->party == sci::ALICE)
    {
#ifdef FFN_LOG
        INIT_TIMER
        START_TIMER
#endif
        double va = dist(gen);
        uint64_t fix_va = sci::neg_mod(static_cast<int64_t>(va * (1ULL << (DEFAULT_ELL))), (1ULL << DEFAULT_ELL));
        bfv_matrix va_xa = conv->he_to_ss_client(io, party);
        conv->Prime_to_Ring(va_xa.data(), va_xa.data(), va_xa.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        FixArray fix_va_xa =
            fpmath->fix->input(sci::ALICE, va_xa.size(), va_xa.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        fix_va_xa = fpmath->fix->mul(fix_va_xa, fix_va);
        fix_va_xa = fpmath->fix->location_truncation(fix_va_xa, DEFAULT_SCALE);
        fix_va_xa.party = sci::PUBLIC;
        FixArray fix_w1 = fpmath->fix->input(sci::ALICE, W1.size(), W1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray va_x_W1a = fpmath->dot(fix_va_xa, fix_w1, batch_size, d_module, ffn_dim, DEFAULT_ELL);
        va_x_W1a = fpmath->fix->location_truncation(va_x_W1a, DEFAULT_SCALE);
        va_x_W1a.party = sci::PUBLIC;
        fix_w1 = fpmath->fix->mul(fix_w1, fix_va);
        fix_w1 = fpmath->fix->location_truncation(fix_w1, DEFAULT_SCALE);
        fix_w1.party = sci::PUBLIC;
        uint64_t prime_div_va = sci::neg_mod(static_cast<int64_t>(va * parm->plain_mod), parm->plain_mod);
        uint64_t *prime_b1 = new uint64_t[b1.size()];
        conv->Ring_to_Prime(b1.data(), prime_b1, b1.size(), DEFAULT_ELL, party->parm->plain_mod);
        BFVLongPlaintext b1_plain(parm, prime_b1, b1.size());
        delete[] prime_b1;
        BFVLongCiphertext b1_secret_a(b1_plain, party), div_va_secret_a(party->parm, prime_div_va, party);
        fpmath->fix->send_fix_array(fix_va_xa);
        fpmath->fix->send_fix_array(va_x_W1a);
        fpmath->fix->send_fix_array(fix_w1);
        BFVLongCiphertext::send(io, &b1_secret_a, true);
        BFVLongCiphertext::send(io, &div_va_secret_a, true);

        // alice receive s0_sgn_secret_a, s1_sgn_secret_a, s2_sgn_secret_a, s3_sgn_secret_a, x1_secret_a
        BFVLongCiphertext s0_sgn_secret_a, s1_sgn_secret_a, s2_sgn_secret_a, s3_sgn_secret_a, x1_secret_a;
        BFVLongCiphertext::recv(io, &s0_sgn_secret_a, parm->context, true);
        BFVLongCiphertext::recv(io, &s1_sgn_secret_a, parm->context, true);
        BFVLongCiphertext::recv(io, &s2_sgn_secret_a, parm->context, true);
        BFVLongCiphertext::recv(io, &s3_sgn_secret_a, parm->context, true);
        BFVLongCiphertext::recv(io, &x1_secret_a, parm->context, true);

        BFVLongPlaintext rb_x1_plain = x1_secret_a.decrypt(party), s0_sgn_plain_a = s0_sgn_secret_a.decrypt(party),
                         s1_sgn_plain_a = s1_sgn_secret_a.decrypt(party),
                         s2_sgn_plain_a = s2_sgn_secret_a.decrypt(party),
                         s3_sgn_plain_a = s3_sgn_secret_a.decrypt(party);
        bfv_matrix ra_rb_x1 = rb_x1_plain.decode_uint(parm), s0_sgn_a = s0_sgn_plain_a.decode_uint(parm),
                   s1_sgn_a = s1_sgn_plain_a.decode_uint(parm), s2_sgn_a = s2_sgn_plain_a.decode_uint(parm),
                   s3_sgn_a = s3_sgn_plain_a.decode_uint(parm);
        double ra = dist(gen);
        uint64_t fix_ra = sci::neg_mod(static_cast<int64_t>(ra * (1ULL << (DEFAULT_ELL))), (1ULL << DEFAULT_ELL)),
                 fix_div_ra = sci::neg_mod(static_cast<int64_t>(1. / ra * parm->plain_mod), parm->plain_mod),
                 fix_div_ra_2 = sci::neg_mod(static_cast<int64_t>(1. / (ra * ra) * parm->plain_mod), parm->plain_mod),
                 fix_div_ra_3 =
                     sci::neg_mod(static_cast<int64_t>(1. / (ra * ra * ra) * parm->plain_mod), parm->plain_mod),
                 fix_div_ra_4 =
                     sci::neg_mod(static_cast<int64_t>(1. / (ra * ra * ra * ra) * parm->plain_mod), parm->plain_mod);
        FixArray fixed_ra_rb_x1 =
            fpmath->fix->input(sci::ALICE, ra_rb_x1.size(), ra_rb_x1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        fixed_ra_rb_x1 = fpmath->fix->mul(fixed_ra_rb_x1, fix_ra);
        fixed_ra_rb_x1 = fpmath->fix->location_truncation(fixed_ra_rb_x1, DEFAULT_SCALE);
        fixed_ra_rb_x1.party = sci::PUBLIC;
        for (size_t i = 0; i < batch_size * ffn_dim; i++)
        {
            s0_sgn_a[i] = s0_sgn_a[i] > parm->plain_mod / 2 ? 1 : parm->plain_mod - 1;
            s1_sgn_a[i] = s1_sgn_a[i] > parm->plain_mod / 2 ? 1 : parm->plain_mod - 1;
            s2_sgn_a[i] = s2_sgn_a[i] > parm->plain_mod / 2 ? 1 : parm->plain_mod - 1;
            s3_sgn_a[i] = s3_sgn_a[i] > parm->plain_mod / 2 ? 1 : parm->plain_mod - 1;
        }

        BFVLongCiphertext div_ra_secret_a(parm, fix_div_ra, party), div_ra_2_secret_a(parm, fix_div_ra_2, party),
            div_ra_3_secret_a(parm, fix_div_ra_3, party), div_ra_4_secret_a(parm, fix_div_ra_4, party);
        // send s0_sgn_a, s1_sgn_a, s2_sgn_a, s3_sgn_a, rs_rc_x1, div_rc_secret_a, div_rc_2_secret_a, div_rc_3_secret_a,
        // div_rc_4_secret_a to bob;
        send_mat(io, &s0_sgn_a);
        send_mat(io, &s1_sgn_a);
        send_mat(io, &s2_sgn_a);
        send_mat(io, &s3_sgn_a);
        fpmath->fix->send_fix_array(fixed_ra_rb_x1);
        BFVLongCiphertext::send(io, &div_ra_secret_a, true);
        BFVLongCiphertext::send(io, &div_ra_2_secret_a, true);
        BFVLongCiphertext::send(io, &div_ra_3_secret_a, true);
        BFVLongCiphertext::send(io, &div_ra_4_secret_a, true);

        bfv_matrix gelu_c = conv->he_to_ss_client(io, party);
        double v1a = dist(gen);
        uint64_t fix_v1a = sci::neg_mod(static_cast<int64_t>(v1a * (1ULL << (DEFAULT_ELL))), (1ULL << DEFAULT_ELL));
        FixArray fix_v1a_x1a =
            fpmath->fix->input(sci::ALICE, gelu_c.size(), gelu_c.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        fix_v1a_x1a = fpmath->fix->mul(fix_v1a_x1a, fix_v1a);
        fix_v1a_x1a = fpmath->fix->location_truncation(fix_v1a_x1a, DEFAULT_SCALE);
        fix_v1a_x1a.party = sci::PUBLIC;
        FixArray fix_w2 = fpmath->fix->input(sci::ALICE, W1.size(), W1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray v1a_x1_W2a = fpmath->dot(fix_v1a_x1a, fix_w2, batch_size, ffn_dim, d_module, DEFAULT_ELL);
        v1a_x1_W2a = fpmath->fix->location_truncation(v1a_x1_W2a, DEFAULT_SCALE);
        v1a_x1_W2a.party = sci::PUBLIC;
        fix_w2 = fpmath->fix->mul(fix_w2, fix_v1a);
        fix_w2 = fpmath->fix->location_truncation(fix_w2, DEFAULT_SCALE);
        fix_w2.party = sci::PUBLIC;
        uint64_t prime_div_v1a = sci::neg_mod(static_cast<int64_t>(v1a * parm->plain_mod), parm->plain_mod);
        uint64_t *prime_b2 = new uint64_t[b2.size()];
        conv->Ring_to_Prime(b2.data(), prime_b2, b2.size(), DEFAULT_ELL, party->parm->plain_mod);
        BFVLongPlaintext b2_plain(parm, prime_b2, b2.size());
        delete[] prime_b2;
        BFVLongCiphertext b2_secret_a(b2_plain, party), div_v1a_secret_a(party->parm, prime_div_v1a, party);

        // send v1a_x1a, v1a_x1a_W2a, W2, B2a_secret_a, div_v1a_secret_a to bob
        fpmath->fix->send_fix_array(fix_v1a_x1a);
        fpmath->fix->send_fix_array(v1a_x1_W2a);
        fpmath->fix->send_fix_array(fix_w2);
        BFVLongCiphertext::send(io, &b2_secret_a, true);
        BFVLongCiphertext::send(io, &div_v1a_secret_a, true);

        BFVLongCiphertext x2_secret_a, vb_secert_b;
        // alice receive x2_secert_a, vb_secret_b;
        BFVLongCiphertext::recv(io, &x2_secret_a, parm->context, true);
        BFVLongCiphertext::recv(io, &vb_secert_b, parm->context, true);
        BFVLongPlaintext x2_plain_ = x2_secret_a.decrypt(party);
        // bfv_matrix x2 = x2_plain_.decode_uint(parm);
        vb_secert_b.add_plain_inplace(x2_plain_, parm->evaluator);
#ifdef FFN_LOG
        STOP_TIMER("Feed Forward")
        total_comm = io->counter - total_comm;
        std::cout << "Feed Forward Send data " << total_comm << " Bytes. \n";
#endif
        return vb_secert_b;
    }
    else
    {
#ifdef FFN_LOG
        INIT_TIMER
        START_TIMER
#endif
        bfv_matrix xb = conv->he_to_ss_server(io, party->parm, input);

        FixArray fix_va_xa(sci::PUBLIC, batch_size * d_module, true, DEFAULT_ELL, DEFAULT_SCALE),
            va_x_W1a(sci::PUBLIC, batch_size * ffn_dim, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_w1a(sci::PUBLIC, d_module * ffn_dim, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext b1_secret_a, div_va_secret_a;
        fpmath->fix->recv_fix_array(fix_va_xa);
        fpmath->fix->recv_fix_array(va_x_W1a);
        fpmath->fix->recv_fix_array(fix_w1a);
        BFVLongCiphertext::recv(io, &b1_secret_a, party->parm->context, true);
        BFVLongCiphertext::recv(io, &div_va_secret_a, party->parm->context, true);
        FixArray fix_w1 = fpmath->fix->input(sci::BOB, W1.size(), W1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_x = fpmath->fix->input(sci::BOB, xb.size(), xb.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray tmp11 = fpmath->dot(fix_va_xa, fix_w1, batch_size, d_module, ffn_dim, DEFAULT_ELL);
        FixArray tmp12 = fpmath->dot(fix_x, fix_w1a, batch_size, d_module, ffn_dim, DEFAULT_ELL);
        FixArray xb_W1b = fpmath->dot(fix_x, fix_w1, batch_size, d_module, ffn_dim, DEFAULT_ELL);
        int ell_mask = tmp11.ell_mask();
        for (size_t i = 0; i < batch_size * ffn_dim; i++)
        {
            tmp11.data[i] = (tmp11.data[i] + tmp12.data[i] + va_x_W1a.data[i]) & ell_mask;
        }
        uint64_t *prime_b1 = new uint64_t[b1.size()];
        conv->Ring_to_Prime(tmp11.data, tmp11.data, tmp11.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(xb_W1b.data, xb_W1b.data, xb_W1b.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(b1.data(), prime_b1, b1.size(), DEFAULT_ELL, parm->plain_mod);
        BFVLongPlaintext tmp11_plain(parm, tmp11.data, tmp11.size), xb_W1b_plain(parm, xb_W1b.data, xb_W1b.size),
            B1b_plain(parm, prime_b1, b1.size());
        delete[] prime_b1;
        BFVLongCiphertext x1_secret_a = div_va_secret_a.multiply_plain(tmp11_plain, parm->evaluator);
        x1_secret_a.add_plain_inplace(xb_W1b_plain, parm->evaluator);
        b1_secret_a.add_plain_inplace(B1b_plain, parm->evaluator);
        x1_secret_a.add_inplace(b1_secret_a, parm->evaluator);

        uint64_t prime_parm0 = sci::neg_mod(static_cast<int64_t>(5.075 * parm->plain_mod), parm->plain_mod),
                 prime_parm1 = sci::neg_mod(static_cast<int64_t>(sqrt(2) * parm->plain_mod), parm->plain_mod),
                 prime_parm2 = sci::neg_mod(static_cast<int64_t>(-sqrt(2) * parm->plain_mod), parm->plain_mod),
                 prime_parm3 = sci::neg_mod(static_cast<int64_t>(-5.075 * parm->plain_mod), parm->plain_mod);
        BFVLongPlaintext parm0(parm, prime_parm0), parm1(parm, prime_parm0), parm2(parm, prime_parm0),
            parm3(parm, prime_parm0);
        BFVLongCiphertext s0_sgn_secret_a = x1_secret_a.add_plain(parm0, parm->evaluator),
                          s1_sgn_secret_a = x1_secret_a.add_plain(parm1, parm->evaluator),
                          s2_sgn_secret_a = x1_secret_a.add_plain(parm2, parm->evaluator),
                          s3_sgn_secret_a = x1_secret_a.add_plain(parm3, parm->evaluator);
        bfv_matrix s0_sgn = random_sgn(batch_size * ffn_dim, parm->plain_mod),
                   s1_sgn = random_sgn(batch_size * ffn_dim, parm->plain_mod),
                   s2_sgn = random_sgn(batch_size * ffn_dim, parm->plain_mod),
                   s3_sgn = random_sgn(batch_size * ffn_dim, parm->plain_mod);
        BFVLongPlaintext s0_sgn_plain_b(parm, s0_sgn.data(), batch_size * ffn_dim),
            s1_sgn_plain_b(parm, s0_sgn.data(), batch_size * ffn_dim),
            s2_sgn_plain_b(parm, s0_sgn.data(), batch_size * ffn_dim),
            s3_sgn_plain_b(parm, s0_sgn.data(), batch_size * ffn_dim);
        s0_sgn_secret_a.multiply_plain_inplace(s0_sgn_plain_b, parm->evaluator);
        s1_sgn_secret_a.multiply_plain_inplace(s1_sgn_plain_b, parm->evaluator);
        s2_sgn_secret_a.multiply_plain_inplace(s2_sgn_plain_b, parm->evaluator);
        s3_sgn_secret_a.multiply_plain_inplace(s3_sgn_plain_b, parm->evaluator);

        double rb = dist(gen);
        uint64_t prime_rb = sci::neg_mod(static_cast<int64_t>(rb * parm->plain_mod), int64_t(parm->plain_mod)),
                 fix_div_rb = sci::neg_mod(static_cast<int64_t>(1. / rb * (1ULL << (DEFAULT_ELL))),
                                           int64_t((1ULL << (DEFAULT_ELL))));
        BFVLongPlaintext rb_plain(parm, prime_rb);
        x1_secret_a.multiply_plain_inplace(rb_plain, parm->evaluator);
        // send s0_sgn_secret_a, s1_sgn_secret_a, s2_sgn_secret_a, s3_sgn_secret_a, x1_secret_a to alice
        BFVLongCiphertext::send(io, &s0_sgn_secret_a, true);
        BFVLongCiphertext::send(io, &s1_sgn_secret_a, true);
        BFVLongCiphertext::send(io, &s2_sgn_secret_a, true);
        BFVLongCiphertext::send(io, &s3_sgn_secret_a, true);
        BFVLongCiphertext::send(io, &x1_secret_a, true);

        bfv_matrix s0_sgn_a(batch_size * ffn_dim), s1_sgn_a(batch_size * ffn_dim), s2_sgn_a(batch_size * ffn_dim),
            s3_sgn_a(batch_size * ffn_dim);
        FixArray fixed_ra_rb_x1(sci::PUBLIC, batch_size * ffn_dim, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext x1_secret_a_, x1_2_secret_a_, x1_3_secret_a_, x1_4_secret_a_;
        recv_mat(io, &s0_sgn_a);
        recv_mat(io, &s1_sgn_a);
        recv_mat(io, &s2_sgn_a);
        recv_mat(io, &s3_sgn_a);
        fpmath->fix->recv_fix_array(fixed_ra_rb_x1);
        BFVLongCiphertext::recv(io, &x1_secret_a_, parm->context, true);
        BFVLongCiphertext::recv(io, &x1_2_secret_a_, parm->context, true);
        BFVLongCiphertext::recv(io, &x1_3_secret_a_, parm->context, true);
        BFVLongCiphertext::recv(io, &x1_4_secret_a_, parm->context, true);
        for (size_t i = 0; i < batch_size * ffn_dim; i++)
        {
            s0_sgn[i] = s0_sgn_a[i] * s0_sgn[i] > parm->plain_mod / 2 ? 1 : 0;
            s1_sgn[i] = s1_sgn_a[i] * s1_sgn[i] > parm->plain_mod / 2 ? 1 : 0;
            s2_sgn[i] = s2_sgn_a[i] * s2_sgn[i] > parm->plain_mod / 2 ? 1 : 0;
            s3_sgn[i] = s3_sgn_a[i] * s3_sgn[i] > parm->plain_mod / 2 ? 1 : 0;
        }
        FixArray fixed_x1 = fpmath->fix->mul(fixed_ra_rb_x1, fix_div_rb);
        fixed_x1 = fpmath->fix->location_truncation(fixed_x1, DEFAULT_SCALE);
        fixed_x1.party = sci::PUBLIC;
        FixArray fixed_x1_2 = fpmath->fix->mul(fixed_x1, fixed_x1, DEFAULT_ELL);
        fixed_x1_2 = fpmath->fix->location_truncation(fixed_x1_2, DEFAULT_SCALE);
        FixArray fixed_x1_3 = fpmath->fix->mul(fixed_x1_2, fixed_x1, DEFAULT_ELL);
        fixed_x1_3 = fpmath->fix->location_truncation(fixed_x1_3, DEFAULT_SCALE);
        FixArray fixed_x1_4 = fpmath->fix->mul(fixed_x1_3, fixed_x1, DEFAULT_ELL);
        fixed_x1_4 = fpmath->fix->location_truncation(fixed_x1_4, DEFAULT_SCALE);
        bfv_matrix b1_(batch_size * ffn_dim), b2_(batch_size * ffn_dim), b3_(batch_size * ffn_dim),
            b4_(batch_size * ffn_dim);
        for (size_t i = 0; i < batch_size * ffn_dim; i++)
        {
            b1_[i] = (s0_sgn[i] == 1 && s1_sgn[i] == 0) ? 1 : 0;
            b2_[i] = (s1_sgn[i] == 1 && s2_sgn[i] == 0) ? 1 : 0;
            b3_[i] = (s2_sgn[i] == 1 && s3_sgn[i] == 0) ? 1 : 0;
            b4_[i] = s3_sgn[i] == 1 ? 1 : 0;
        }
        conv->Ring_to_Prime(fixed_x1.data, fixed_x1.data, fixed_x1.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(fixed_x1_2.data, fixed_x1_2.data, fixed_x1.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(fixed_x1_3.data, fixed_x1_3.data, fixed_x1.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(fixed_x1_4.data, fixed_x1_4.data, fixed_x1.size, DEFAULT_ELL, parm->plain_mod);
        BFVLongPlaintext x1_plain(parm, fixed_x1.data, fixed_x1.size),
            x1_2_plain(parm, fixed_x1_2.data, fixed_x1_2.size), x1_3_plain(parm, fixed_x1_3.data, fixed_x1_3.size),
            x1_4_plain(parm, fixed_x1_4.data, fixed_x1_4.size);
        x1_secret_a_.multiply_plain_inplace(x1_plain, parm->evaluator);
        x1_2_secret_a_.multiply_plain_inplace(x1_2_plain, parm->evaluator);
        x1_3_secret_a_.multiply_plain_inplace(x1_3_plain, parm->evaluator);
        x1_4_secret_a_.multiply_plain_inplace(x1_4_plain, parm->evaluator);
        BFVLongPlaintext b1_plain_(parm, b1_), b2_plain_(parm, b2_), b3_plain_(parm, b3_), b4_plain_(parm, b4_);
        BFVLongCiphertext f4_ = x1_secret_a_.multiply_plain(b4_plain_, parm->evaluator),
                          f3_ = f3(x1_secret_a_, x1_2_secret_a_, x1_3_secret_a_),
                          f2_ = f2(x1_secret_a_, x1_2_secret_a_, x1_4_secret_a_),
                          f1_ = f1(x1_secret_a_, x1_2_secret_a_, x1_3_secret_a_, x1_4_secret_a_);
        f3_.multiply_plain_inplace(b3_plain_, parm->evaluator);
        f2_.multiply_plain_inplace(b2_plain_, parm->evaluator);
        f1_.multiply_plain_inplace(b1_plain_, parm->evaluator);
        f1_.add_inplace(f3_, parm->evaluator);
        f1_.add_inplace(f2_, parm->evaluator);
        f4_.mod_switch_to_inplace(f1_.parms_id(), parm->evaluator);
        f1_.add_inplace(f4_, parm->evaluator); // f1_ is gelu
        bfv_matrix gelu_s = conv->he_to_ss_server(io, parm, f1_);

        FixArray fix_v1a_x1a(sci::PUBLIC, batch_size * ffn_dim, true, DEFAULT_ELL, DEFAULT_SCALE),
            v1a_x1_W2a(sci::PUBLIC, batch_size * d_module, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_w2a(sci::PUBLIC, ffn_dim * d_module, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext b2_secret_a, div_v1a_secret_a;
        fpmath->fix->recv_fix_array(fix_v1a_x1a);
        fpmath->fix->recv_fix_array(v1a_x1_W2a);
        fpmath->fix->recv_fix_array(fix_w2a);
        BFVLongCiphertext::recv(io, &b2_secret_a, party->parm->context, true);
        BFVLongCiphertext::recv(io, &div_v1a_secret_a, party->parm->context, true);
        FixArray fix_w2 = fpmath->fix->input(sci::BOB, W2.size(), W2.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_x1 = fpmath->fix->input(sci::BOB, gelu_s.size(), gelu_s.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray tmp21 = fpmath->dot(fix_v1a_x1a, fix_w2, batch_size, ffn_dim, d_module, DEFAULT_ELL);
        FixArray tmp22 = fpmath->dot(fix_x1, fix_w2a, batch_size, ffn_dim, d_module, DEFAULT_ELL);
        FixArray x1b_W2b = fpmath->dot(fix_x1, fix_w2, batch_size, ffn_dim, d_module, DEFAULT_ELL);
        ell_mask = tmp21.ell_mask();
        for (size_t i = 0; i < batch_size * d_module; i++)
        {
            tmp21.data[i] = (tmp21.data[i] + tmp22.data[i] + v1a_x1_W2a.data[i]) & ell_mask;
        }
        uint64_t *prime_b2 = new uint64_t[b2.size()];
        conv->Ring_to_Prime(tmp21.data, tmp21.data, tmp21.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(x1b_W2b.data, x1b_W2b.data, x1b_W2b.size, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(b2.data(), prime_b2, b2.size(), DEFAULT_ELL, parm->plain_mod);
        BFVLongPlaintext tmp21_plain(parm, tmp21.data, tmp21.size), x1b_W2b_plain(parm, x1b_W2b.data, x1b_W2b.size),
            B2b_plain(parm, prime_b2, b2.size());
        delete[] prime_b2;
        BFVLongCiphertext x2_secret_a = div_v1a_secret_a.multiply_plain(tmp21_plain, parm->evaluator);
        x2_secret_a.add_plain_inplace(x1b_W2b_plain, parm->evaluator);
        b2_secret_a.add_plain_inplace(B2b_plain, parm->evaluator);
        x2_secret_a.add_inplace(b2_secret_a, parm->evaluator);

        double vb = dist(gen);
        uint64_t prime_vb = sci::neg_mod(static_cast<int64_t>(vb * parm->plain_mod), int64_t(parm->plain_mod)),
                 prime_div_vb = sci::neg_mod(static_cast<int64_t>(1. / vb * parm->plain_mod), int64_t(parm->plain_mod));
        BFVLongPlaintext div_vb_plain(parm, prime_div_vb);
        x2_secret_a.multiply_plain_inplace(div_vb_plain, parm->evaluator);
        BFVLongCiphertext vb_secret_b(parm, prime_vb, party);
        // send x2_secret_a, vs_secret_b to alice
        BFVLongCiphertext::send(io, &x2_secret_a, true);
        BFVLongCiphertext::send(io, &vb_secret_b, true);
#ifdef FFN_LOG
        STOP_TIMER("Feed Forward")
        total_comm = io->counter - total_comm;
        std::cout << "Feed Forward Send data " << total_comm << " Bytes. \n";
#endif
    }
    return BFVLongCiphertext();
}