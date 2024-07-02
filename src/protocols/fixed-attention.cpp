#include "fixed-attention.h"
#include "FixedPoint/fixed-point.h"
#include "Utils/constants.h"
#include "model.h"
#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include <cstddef>

Fixed_Attention::Fixed_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath,
                                 FPMath *fpmath_public, Conversion *conv, int head_)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv), head(head_) {}

bfv_matrix Fixed_Attention::forward(const bfv_matrix &input) const
{

    sci::PRG128 prg;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);
    int total_commun = 0;
    uint64_t *x = new uint64_t[input.size()];
    for (size_t i = 0; i < input.size(); i++)
    {
        x[i] = input[i];
    }

    if (party->party == sci::ALICE)
    {
        double ra = dist(gen);
        uint64_t fix_ra = sci::neg_mod(static_cast<int64_t>(ra * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        FixArray fix_xa = fpmath->fix->input(sci::ALICE, batch_size * d_module, x, true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_wq = fpmath->fix->input(sci::ALICE, d_module * d_k, WQ.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_wk = fpmath->fix->input(sci::ALICE, d_module * d_k, WK.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_wv = fpmath->fix->input(sci::ALICE, d_module * d_k, WV.data(), true, DEFAULT_ELL, DEFAULT_SCALE);

        FixArray fix_ra_xa = fpmath->fix->mul(fix_xa, fix_ra, DEFAULT_ELL),
                 fix_ra_wq = fpmath->fix->mul(fix_wq, fix_ra, DEFAULT_ELL),
                 fix_ra_wk = fpmath->fix->mul(fix_wk, fix_ra, DEFAULT_ELL),
                 fix_ra_wv = fpmath->fix->mul(fix_wv, fix_ra, DEFAULT_ELL);
        fix_ra_xa = fpmath->fix->location_truncation(fix_ra_xa, DEFAULT_SCALE);
        fix_ra_wq = fpmath->fix->location_truncation(fix_ra_wq, DEFAULT_SCALE);
        fix_ra_wk = fpmath->fix->location_truncation(fix_ra_wk, DEFAULT_SCALE);
        fix_ra_wv = fpmath->fix->location_truncation(fix_ra_wv, DEFAULT_SCALE);

        fix_ra_xa.party = sci::PUBLIC;
        fix_ra_wq.party = sci::PUBLIC;
        fix_ra_wk.party = sci::PUBLIC;
        fix_ra_wv.party = sci::PUBLIC;

        FixArray ra_xa_WQa = fpmath->dot(fix_xa, fix_ra_wq, batch_size, d_module, d_k, DEFAULT_ELL);
        FixArray ra_xa_WKa = fpmath->dot(fix_xa, fix_ra_wk, batch_size, d_module, d_k, DEFAULT_ELL);
        FixArray ra_xa_WVa = fpmath->dot(fix_xa, fix_ra_wv, batch_size, d_module, d_k, DEFAULT_ELL);
        uint64_t ell_mask_ = ra_xa_WQa.ell_mask();
        FixArray fix_bq = fpmath->fix->input(sci::ALICE, bQ.size(), bQ.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_bk = fpmath->fix->input(sci::ALICE, bK.size(), bK.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_bv = fpmath->fix->input(sci::ALICE, bV.size(), bV.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        for (size_t i = 0; i < batch_size; i++)
        {
            for (size_t j = 0; j < d_k; j++)
            {
                ra_xa_WQa.data[i * d_k + j] += fix_bq.data[j];
                ra_xa_WQa.data[i * d_k + j] &= ell_mask_;
                ra_xa_WKa.data[i * d_k + j] += fix_bq.data[j];
                ra_xa_WKa.data[i * d_k + j] &= ell_mask_;
                ra_xa_WVa.data[i * d_k + j] += fix_bq.data[j];
                ra_xa_WVa.data[i * d_k + j] &= ell_mask_;
            }
        }
        BFVLongCiphertext ra_secret_a(parm, fix_ra, party);
        ra_xa_WQa.party = sci::PUBLIC;
        ra_xa_WKa.party = sci::PUBLIC;
        ra_xa_WVa.party = sci::PUBLIC;
        fpmath->fix->send_fix_array(ra_xa_WQa);
        fpmath->fix->send_fix_array(ra_xa_WKa);
        fpmath->fix->send_fix_array(ra_xa_WVa);
        fpmath->fix->send_fix_array(fix_ra_xa);
        fpmath->fix->send_fix_array(fix_ra_wq);
        fpmath->fix->send_fix_array(fix_ra_wk);
        fpmath->fix->send_fix_array(fix_ra_wv);
        BFVLongCiphertext::send(io, &ra_secret_a);

        BFVLongCiphertext raQ_sec_a, raK_sec_a, rb1_square_secret_b;
        BFVLongCiphertext::recv(io, &raQ_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &raK_sec_a, party->parm->context);
        BFVLongCiphertext::recv(io, &rb1_square_secret_b, party->parm->context);

        BFVLongPlaintext raQ_div_rb1_plain = raQ_sec_a.decrypt(party);
        BFVLongPlaintext raK_div_rb1_plain = raK_sec_a.decrypt(party);
        bfv_matrix Q_div_rb1 = raQ_div_rb1_plain.decode(parm);
        bfv_matrix K_div_rb1 = raK_div_rb1_plain.decode(parm);
        bfv_matrix eScore_a(batch_size * batch_size);
        random_ell_mat(eScore_a, DEFAULT_ELL);
        double sqrt_d_k = sqrt(d_k);
        // uint64_t *Q_div_rb1_ring = new uint64_t[Q_div_rb1.size()];
        // uint64_t *K_div_rb1_ring = new uint64_t[K_div_rb1.size()];
        conv->Prime_to_Ring(Q_div_rb1.data(), Q_div_rb1.data(), Q_div_rb1.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        conv->Prime_to_Ring(K_div_rb1.data(), K_div_rb1.data(), K_div_rb1.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        FixArray fix_Q_div_rb1 =
            fpmath->fix->input(sci::PUBLIC, Q_div_rb1.size(), Q_div_rb1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_K_div_rb1 =
            fpmath->fix->input(sci::PUBLIC, K_div_rb1.size(), K_div_rb1.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        uint64_t fix_div_ra =
            sci::neg_mod(static_cast<int64_t>(1. / ra * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        uint64_t fix_div_sqrt_d_k =
            sci::neg_mod(static_cast<int64_t>(1. / sqrt_d_k * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        fix_Q_div_rb1 = fpmath->fix->mul(fix_Q_div_rb1, fix_div_ra);
        fix_Q_div_rb1 = fpmath->fix->mul(fix_Q_div_rb1, fix_div_sqrt_d_k);
        fix_K_div_rb1 = fpmath->fix->mul(fix_K_div_rb1, fix_div_ra);
        fix_Q_div_rb1.party = sci::ALICE;
        FixArray temp_Score = fpmath->dot(fix_Q_div_rb1, fix_K_div_rb1, batch_size, d_k, batch_size, DEFAULT_ELL, true);
        // Alice End
    }
    else
    {
        FixArray ra_xa_WQa(sci::PUBLIC, batch_size * d_k, true, DEFAULT_ELL, DEFAULT_SCALE),
            ra_xa_WKa(sci::PUBLIC, batch_size * d_k, true, DEFAULT_ELL, DEFAULT_SCALE),
            ra_xa_WVa(sci::PUBLIC, batch_size * d_k, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_ra_xa(sci::PUBLIC, batch_size * d_module, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_ra_wqa(sci::PUBLIC, d_module * d_k, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_ra_wka(sci::PUBLIC, d_module * d_k, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_ra_wva(sci::PUBLIC, d_module * d_k, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext ra_secret_a;
        fpmath->fix->recv_fix_array(ra_xa_WQa);
        fpmath->fix->recv_fix_array(ra_xa_WKa);
        fpmath->fix->recv_fix_array(ra_xa_WVa);
        fpmath->fix->recv_fix_array(fix_ra_xa);
        fpmath->fix->recv_fix_array(fix_ra_wqa);
        fpmath->fix->recv_fix_array(fix_ra_wka);
        fpmath->fix->recv_fix_array(fix_ra_wva);
        BFVLongCiphertext::recv(io, &ra_secret_a, party->parm->context);

        auto cal_raI = [this, &fix_ra_xa, &ra_secret_a](FixArray &fix_input, FixArray &ra_xa_WIa, const bfv_matrix &WIb,
                                                        FixArray &ra_WIa, const bfv_matrix &bI)
        {
            FixArray fix_wib = fpmath->fix->input(sci::BOB, WIb.size(), WIb.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
            FixArray xb_wib = fpmath->dot(fix_input, fix_wib, batch_size, d_module, d_k, DEFAULT_ELL);
            uint64_t *prime_xb_wib = new uint64_t[d_module * d_k];
            conv->Ring_to_Prime(xb_wib.data, prime_xb_wib, xb_wib.size, DEFAULT_ELL, party->parm->plain_mod);
            BFVLongPlaintext xb_WIb_plain(party->parm, prime_xb_wib, xb_wib.size);
            BFVLongCiphertext raI_secret_a = ra_secret_a.multiply_plain(xb_WIb_plain, party->parm->evaluator);
            uint64_t *temp_raI = new uint64_t[batch_size * d_k];
            FixArray temp_raI1 = fpmath->dot(fix_ra_xa, fix_wib, batch_size, d_module, d_k, DEFAULT_ELL);
            FixArray temp_raI2 = fpmath->dot(fix_input, ra_WIa, batch_size, d_module, d_k, DEFAULT_ELL);
            uint64_t ell_mask_ = temp_raI1.ell_mask();
            for (size_t i = 0; i < batch_size; i++)
            {
                for (size_t j = 0; j < d_k; j++)
                {
                    temp_raI[i * d_k + j] =
                        ra_xa_WIa.data[i * d_k + j] + temp_raI1.data[i * d_k + j] + temp_raI2.data[i * d_k + j] + bI[j];
                    temp_raI[i * d_k + j] &= ell_mask_;
                }
            }
            conv->Ring_to_Prime(temp_raI, temp_raI, batch_size * d_k, DEFAULT_ELL, party->parm->plain_mod);
            BFVLongPlaintext temp_raI_plain(party->parm, temp_raI, batch_size * d_k);
            raI_secret_a.add_plain_inplace(temp_raI_plain, party->parm->evaluator);
            delete[] temp_raI;
            delete[] prime_xb_wib;
            return raI_secret_a;
        };
        FixArray fix_input = fpmath->fix->input(sci::BOB, input.size(), input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext raQ_sec_a = cal_raI(fix_input, ra_xa_WQa, WQ, fix_ra_wqa, bQ);
        BFVLongCiphertext raK_sec_a = cal_raI(fix_input, ra_xa_WKa, WK, fix_ra_wka, bK);
        BFVLongCiphertext raV_sec_a = cal_raI(fix_input, ra_xa_WVa, WV, fix_ra_wva, bV);

        double rb1 = dist(gen);
        uint64_t div_fix_rb1 =
            sci::neg_mod(static_cast<int64_t>(1. / rb1 * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        uint64_t fix_rb1_square =
            sci::neg_mod(static_cast<int64_t>(rb1 * rb1 * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        BFVLongPlaintext div_rb1_plain(
            party->parm, sci::neg_mod(sci::signed_val(div_fix_rb1, DEFAULT_ELL), (int64_t)party->parm->plain_mod));
        BFVLongCiphertext rb1_square_secret_b(
            party->parm, sci::neg_mod(sci::signed_val(fix_rb1_square, DEFAULT_ELL), (int64_t)party->parm->plain_mod),
            party);
        raQ_sec_a.multiply_plain_inplace(div_rb1_plain, party->parm->evaluator);
        raK_sec_a.multiply_plain_inplace(div_rb1_plain, party->parm->evaluator);

        BFVLongCiphertext::send(io, &raQ_sec_a);
        BFVLongCiphertext::send(io, &raK_sec_a);
        BFVLongCiphertext::send(io, &rb1_square_secret_b);
    }
    return bfv_matrix(batch_size * d_k, 1);
}

Fixed_Multi_Head_Attention::Fixed_Multi_Head_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io,
                                                       FPMath *fpmath, FPMath *fpmath_public, Conversion *conv)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv)
{
    attns = new Fixed_Attention *[n_heads];
    string layer_str = std::to_string(layer),
           WQ_file = replace("bert.encoder.layer.LAYER.attention.self.query.weight.txt", "LAYER", layer_str),
           WK_file = replace("bert.encoder.layer.LAYER.attention.self.key.weight.txt", "LAYER", layer_str),
           WV_file = replace("bert.encoder.layer.LAYER.attention.self.value.weight.txt", "LAYER", layer_str),
           bQ_file = replace("bert.encoder.layer.LAYER.attention.self.query.bias.txt", "LAYER", layer_str),
           bK_file = replace("bert.encoder.layer.LAYER.attention.self.key.bias.txt", "LAYER", layer_str),
           bV_file = replace("bert.encoder.layer.LAYER.attention.self.value.bias.txt", "LAYER", layer_str);
    bfv_matrix allWQ, allWK, allWV, bQ, bK, bV;
    load_bfv_mat(allWQ, dir_path + WQ_file);
    load_bfv_mat(allWK, dir_path + WK_file);
    load_bfv_mat(allWV, dir_path + WV_file);
    load_bfv_mat(bQ, dir_path + bQ_file);
    load_bfv_mat(bK, dir_path + bK_file);
    load_bfv_mat(bV, dir_path + bV_file);
    size_t size = d_module * d_k;
    for (int i = 0; i < n_heads; i++)
    {
        attns[i] = new Fixed_Attention(layer, party, parm, io, fpmath, fpmath_public, conv, i);
        attns[i]->WQ = bfv_matrix(allWQ.begin() + i * size, allWQ.begin() + (i + 1) * size);
        attns[i]->WK = bfv_matrix(allWK.begin() + i * size, allWK.begin() + (i + 1) * size);
        attns[i]->WV = bfv_matrix(allWV.begin() + i * size, allWV.begin() + (i + 1) * size);
        attns[i]->bQ = bfv_matrix(bQ.begin() + i * d_k, bQ.begin() + (i + 1) * d_k);
        attns[i]->bK = bfv_matrix(bK.begin() + i * d_k, bK.begin() + (i + 1) * d_k);
        attns[i]->bV = bfv_matrix(bV.begin() + i * d_k, bV.begin() + (i + 1) * d_k);
    }
}

Fixed_Multi_Head_Attention::~Fixed_Multi_Head_Attention()
{
    for (int i = 0; i < n_heads; i++)
    {
        delete attns[i];
    }
    delete[] attns;
}

BFVLongCiphertext Fixed_Multi_Head_Attention::forward(const bfv_matrix &input) const
{
    bfv_matrix output(batch_size * d_module);

    size_t i, j;
    for (int h = 0; h < n_heads; h++)
    {
        bfv_matrix output_h = attns[h]->forward(input);
        for (i = 0; i < batch_size; i++)
        {
            for (j = 0; j < d_k; j++)
            {
                output[i * d_module + h * d_k + j] = output_h[i * d_k + j];
            }
        }
    }

    BFVLongCiphertext output_secret;
    if (party->party == sci::ALICE)
    {
        BFVLongCiphertext::recv(io, &output_secret, party->parm->context);
        BFVLongPlaintext output_plain = BFVLongPlaintext(party->parm, output);
        output_secret.multiply_plain_inplace(output_plain, party->parm->evaluator);
    }
    else
    {
        BFVLongCiphertext output_secret_b(BFVLongPlaintext(party->parm, output), party);
        BFVLongCiphertext::send(io, &output_secret_b);
    }
    return output_secret;
}