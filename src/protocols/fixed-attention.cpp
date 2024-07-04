#include "fixed-attention.h"
#include "FixedPoint/fixed-point.h"
#include "Utils/constants.h"
#include "model.h"
#include "protocols/fixed-protocol.h"
#include "utils.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <cstdint>

Fixed_Attention::Fixed_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath,
                                 FPMath *fpmath_public, Conversion *conv, int head_)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv), head(head_) {}

bfv_matrix Fixed_Attention::forward(const bfv_matrix &input) const {
    size_t total_comm = io->counter;
    sci::PRG128 prg;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);

#ifdef LOG
    INIT_TIMER
    START_TIMER
#endif
    if (party->party == sci::ALICE) {
        double ra = dist(gen);
        uint64_t fix_ra = sci::neg_mod(static_cast<int64_t>(ra * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        FixArray fix_xa =
            fpmath->fix->input(sci::ALICE, batch_size * d_module, input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
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
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < d_k; j++) {
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
        uint64_t *tmp_score_prime = new uint64_t[temp_Score.size];
        conv->Ring_to_Prime(temp_Score.data, tmp_score_prime, temp_Score.size, DEFAULT_ELL, party->parm->plain_mod);
        BFVLongPlaintext Score_plain(parm, tmp_score_prime, temp_Score.size);
        delete[] tmp_score_prime;
        BFVLongCiphertext Score_b_secret_b = rb1_square_secret_b.multiply_plain(Score_plain, party->parm->evaluator);
        bfv_matrix Score_a = conv->he_to_ss_server(io, party->parm, Score_b_secret_b);
        conv->Prime_to_Ring(Score_a.data(), Score_a.data(), Score_a.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        FixArray fix_score_a =
            fpmath->fix->input(sci::PUBLIC, Score_a.size(), Score_a.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_exp_score_a = fpmath->location_exp(fix_score_a, DEFAULT_SCALE, DEFAULT_SCALE);
        uint64_t *fix_exp_score_a_prime = new uint64_t[fix_exp_score_a.size];
        conv->Ring_to_Prime(fix_exp_score_a.data, fix_exp_score_a_prime, fix_exp_score_a.size, DEFAULT_ELL,
                            party->parm->plain_mod);
        BFVLongPlaintext exp_Score_a_plain(parm, fix_exp_score_a_prime, fix_exp_score_a.size);
        delete[] fix_exp_score_a_prime;
        BFVLongCiphertext exp_Score_a_secret_a(exp_Score_a_plain, party);
        BFVLongCiphertext::send(io, &exp_Score_a_secret_a);

        BFVLongCiphertext eScore_a_secret_a, raV_sec_a;
        FixArray fix_exp_score_b(sci::PUBLIC, batch_size * batch_size, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext::recv(io, &eScore_a_secret_a, party->parm->context);
        fpmath->fix->recv_fix_array(fix_exp_score_b);
        BFVLongCiphertext::recv(io, &raV_sec_a, party->parm->context);

        BFVLongPlaintext rs2_expScore_plain = eScore_a_secret_a.decrypt(party);
        bfv_matrix rs2_expScore = rs2_expScore_plain.decode(parm);
        conv->Prime_to_Ring(rs2_expScore.data(), rs2_expScore.data(), rs2_expScore.size(), DEFAULT_ELL,
                            party->parm->plain_mod, DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        FixArray fix_rs2_exp_score =
            fpmath->fix->input(sci::PUBLIC, rs2_expScore.size(), rs2_expScore.data(), true, DEFAULT_ELL, DEFAULT_SCALE);

        BFVLongPlaintext Rb_V_plain = raV_sec_a.decrypt(party);
        bfv_matrix Rb_V = Rb_V_plain.decode(parm);
        conv->Prime_to_Ring(Rb_V.data(), Rb_V.data(), Rb_V.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE,
                            DEFAULT_SCALE, fpmath);
        FixArray fix_rb_v = fpmath->fix->input(sci::PUBLIC, Rb_V.size(), Rb_V.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        fix_rb_v = fpmath->fix->mul(fix_rb_v, fix_div_ra);

        vector<FixArray> fix_rs2_exp_score_vec(batch_size,
                                               FixArray(sci::ALICE, batch_size, true, DEFAULT_ELL, DEFAULT_SCALE));
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < batch_size; j++) {
                fix_rs2_exp_score_vec[i].data[j] = fix_rs2_exp_score.data[i * batch_size + j];
            }
        }
        FixArray exp_sum_ = fpmath->fix->tree_sum(fix_rs2_exp_score_vec);
        FixArray exp_sum(sci::PUBLIC, batch_size * batch_size, true, DEFAULT_ELL, DEFAULT_SCALE);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 1; j < batch_size; j++) {
                exp_sum.data[i * batch_size + j] = exp_sum_.data[i];
            }
        }
        fix_exp_score_a.party = sci::ALICE;
        fix_exp_score_b = fpmath->fix->mul(fix_exp_score_b, fix_exp_score_a, DEFAULT_ELL);
        fix_exp_score_b = fpmath->fix->location_truncation(fix_exp_score_b, DEFAULT_SCALE);
        fix_exp_score_b = fpmath->fix->local_div(fix_exp_score_b, exp_sum);
        fix_rb_v.party = sci::ALICE;
        FixArray fix_output = fpmath->dot(fix_exp_score_b, fix_rb_v, batch_size, batch_size, d_k, DEFAULT_ELL);

        // Alice End
#ifdef LOG
        char *buf = new char[13];
        sprintf(buf, "Attention-%-2d", head);
        STOP_TIMER(buf)
        total_comm = io->counter - total_comm;
        printf("%s Send data %ld Bytes. \n", buf, total_comm);
        delete[] buf;
#endif
        return bfv_matrix(fix_output.data, fix_output.data + fix_output.size);
    } else {
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
                                                        FixArray &ra_WIa, const bfv_matrix &bI) {
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
            for (size_t i = 0; i < batch_size; i++) {
                for (size_t j = 0; j < d_k; j++) {
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

        BFVLongCiphertext exp_Score_a_secret_a;
        bfv_matrix Score_b = conv->he_to_ss_client(io, party);
        conv->Prime_to_Ring(Score_b.data(), Score_b.data(), Score_b.size(), DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
        BFVLongCiphertext::recv(io, &exp_Score_a_secret_a, party->parm->context);

        double rb2 = dist(gen);
        uint64_t fix_rb2 = sci::neg_mod(static_cast<int64_t>(rb2 * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        uint64_t fix_div_rb2 =
            sci::neg_mod(static_cast<int64_t>(1. / rb2 * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        bfv_matrix Db(batch_size * d_k), Rb(batch_size * d_k);
        random_ell_mat(Db, DEFAULT_ELL);
        random_modP_mat(Rb, party->parm->plain_mod);
        for (size_t i = 1; i < batch_size; i++) {
            for (size_t j = 0; j < d_k; j++) {
                Db[i * d_k + j] = Db[j];
                Rb[j * batch_size + i] = Rb[i];
            }
        }
        FixArray fix_score_b =
            fpmath->fix->input(sci::PUBLIC, Score_b.size(), Score_b.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_exp_score_b = fpmath->location_exp(fix_score_b, DEFAULT_SCALE, DEFAULT_SCALE);
        fix_exp_score_b = fpmath->fix->mul(fix_exp_score_b, fix_rb2);
        BFVLongPlaintext rb2_expZb_plain(parm, fix_exp_score_b.data, fix_exp_score_b.size);
        exp_Score_a_secret_a.multiply_plain_inplace(rb2_expZb_plain, party->parm->evaluator);

        FixArray O = fpmath->zero_sum_modP(batch_size, batch_size, party->parm->plain_mod, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongPlaintext O_plain(parm, O.data, O.size);
        exp_Score_a_secret_a.add_plain_inplace(O_plain, party->parm->evaluator);

        uint64_t mask = fix_exp_score_b.ell_mask();
        for (size_t i = 0; i < batch_size * batch_size; i++) {
            fix_exp_score_b.data[i] = (fix_exp_score_b.data[i] * Db[i / batch_size]) & mask;
        }
        fix_exp_score_b = fpmath->fix->location_truncation(fix_exp_score_b, DEFAULT_SCALE);
        fix_exp_score_b = fpmath->fix->mul(fix_exp_score_b, fix_div_rb2);
        BFVLongPlaintext Rb_plain(parm, Rb);
        raV_sec_a.multiply_plain_inplace(Rb_plain, party->parm->evaluator);
        conv->Prime_to_Ring(Rb.data(), Rb.data(), Rb.size(), DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE,
                            DEFAULT_SCALE, fpmath);
        FixArray fix_db = fpmath->fix->input(sci::PUBLIC, Db.size(), Db.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_rb = fpmath->fix->input(sci::PUBLIC, Rb.size(), Rb.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray db_rb = fpmath->fix->mul(fix_db, fix_rb, DEFAULT_ELL);
        db_rb = fpmath->fix->location_truncation(db_rb, DEFAULT_SCALE);
        FixArray fix_rb2_long =
            fpmath->fix->input(sci::BOB, batch_size * d_k, fix_rb2, true, DEFAULT_ELL, DEFAULT_SCALE);
        db_rb.party = sci::PUBLIC;
        FixArray fix_output = fpmath->fix->local_div(fix_rb2_long, db_rb);
        // send H4 = {eScore_a_secret_a, eScore_b, raV_sec_a} to alice
        BFVLongCiphertext::send(io, &exp_Score_a_secret_a);
        fpmath->fix->send_fix_array(fix_exp_score_b);
        BFVLongCiphertext::send(io, &raV_sec_a);
#ifdef LOG
        char *buf = new char[13];
        sprintf(buf, "Attention-%-2d", head);
        STOP_TIMER(buf)
        total_comm = io->counter - total_comm;
        printf("%s Send data %ld Bytes. \n", buf, total_comm);
        delete[] buf;
#endif
        return bfv_matrix(fix_output.data, fix_output.data + fix_output.size);
    }
    return bfv_matrix(batch_size * d_k, 1);
}

Fixed_Multi_Head_Attention::Fixed_Multi_Head_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io,
                                                       FPMath *fpmath, FPMath *fpmath_public, Conversion *conv)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv) {
    attns = new Fixed_Attention *[n_heads];
    string WQ_file = replace("bert.encoder.layer.LAYER.attention.self.query.weight.txt", "LAYER", layer_str),
           WK_file = replace("bert.encoder.layer.LAYER.attention.self.key.weight.txt", "LAYER", layer_str),
           WV_file = replace("bert.encoder.layer.LAYER.attention.self.value.weight.txt", "LAYER", layer_str),
           bQ_file = replace("bert.encoder.layer.LAYER.attention.self.query.bias.txt", "LAYER", layer_str),
           bK_file = replace("bert.encoder.layer.LAYER.attention.self.key.bias.txt", "LAYER", layer_str),
           bV_file = replace("bert.encoder.layer.LAYER.attention.self.value.bias.txt", "LAYER", layer_str),
           W_file = replace("bert.encoder.layer.LAYER.attention.output.dense.weight.txt", "LAYER", layer_str),
           b_file = replace("bert.encoder.layer.LAYER.attention.output.dense.bias.txt", "LAYER", layer_str);
    bfv_matrix allWQ, allWK, allWV, bQ, bK, bV;
    load_bfv_mat(allWQ, dir_path + WQ_file);
    load_bfv_mat(allWK, dir_path + WK_file);
    load_bfv_mat(allWV, dir_path + WV_file);
    load_bfv_mat(bQ, dir_path + bQ_file);
    load_bfv_mat(bK, dir_path + bK_file);
    load_bfv_mat(bV, dir_path + bV_file);
    load_bfv_mat(W, dir_path + W_file);
    load_bfv_mat(b, dir_path + b_file);
    size_t size = d_module * d_k;
    for (int i = 0; i < n_heads; i++) {
        attns[i] = new Fixed_Attention(layer, party, parm, io, fpmath, fpmath_public, conv, i);
        attns[i]->WQ = bfv_matrix(allWQ.begin() + i * size, allWQ.begin() + (i + 1) * size);
        attns[i]->WK = bfv_matrix(allWK.begin() + i * size, allWK.begin() + (i + 1) * size);
        attns[i]->WV = bfv_matrix(allWV.begin() + i * size, allWV.begin() + (i + 1) * size);
        attns[i]->bQ = bfv_matrix(bQ.begin() + i * d_k, bQ.begin() + (i + 1) * d_k);
        attns[i]->bK = bfv_matrix(bK.begin() + i * d_k, bK.begin() + (i + 1) * d_k);
        attns[i]->bV = bfv_matrix(bV.begin() + i * d_k, bV.begin() + (i + 1) * d_k);
    }
}

Fixed_Multi_Head_Attention::~Fixed_Multi_Head_Attention() {
    for (int i = 0; i < n_heads; i++) {
        delete attns[i];
    }
    delete[] attns;
}

BFVLongCiphertext Fixed_Multi_Head_Attention::forward(const bfv_matrix &input) const {
    bfv_matrix output(batch_size * d_module);
#ifdef LOG
    INIT_TIMER
    START_TIMER
#endif
    size_t total_comm = io->counter;
    size_t i, j;
    for (int h = 0; h < n_heads; h++) {
        bfv_matrix output_h = attns[h]->forward(input);
        for (i = 0; i < batch_size; i++) {
            for (j = 0; j < d_k; j++) {
                output[i * d_module + h * d_k + j] = output_h[i * d_k + j];
            }
        }
    }

    if (party->party == sci::ALICE) {
        FixArray fix_output_b(sci::PUBLIC, batch_size * d_module, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_weight_b(sci::PUBLIC, d_module * d_module, true, DEFAULT_ELL, DEFAULT_SCALE),
            fix_attn_output_b(sci::PUBLIC, batch_size * d_module, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext rb_secret_b;
        fpmath->fix->recv_fix_array(fix_output_b);
        fpmath->fix->recv_fix_array(fix_weight_b);
        fpmath->fix->recv_fix_array(fix_attn_output_b);
        BFVLongCiphertext::recv(io, &rb_secret_b, party->parm->context);

        FixArray fix_output_ =
            fpmath->fix->input(sci::ALICE, output.size(), output.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_weight = fpmath->fix->input(sci::PUBLIC, W.size(), W.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_attn_output = fpmath->dot(fix_output_, fix_weight, batch_size, d_module, d_module, DEFAULT_ELL);
        fix_attn_output = fpmath->fix->location_truncation(fix_attn_output, DEFAULT_SCALE);

        FixArray tmp1 = fpmath->dot(fix_output_, fix_weight_b, batch_size, d_module, d_module, DEFAULT_ELL);
        FixArray tmp2 = fpmath->dot(fix_output_b, fix_weight, batch_size, d_module, d_module, DEFAULT_ELL);
        tmp1 = fpmath->fix->location_truncation(tmp1, DEFAULT_SCALE);
        tmp2 = fpmath->fix->location_truncation(tmp2, DEFAULT_SCALE);
        uint64_t ell_mask_ = fix_attn_output.ell_mask();
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < d_module; j++) {
                fix_attn_output.data[i * d_module + j] = fix_attn_output.data[i * d_module + j] +
                                                         tmp1.data[i * d_module + j] + tmp2.data[i * d_module + j] +
                                                         fix_attn_output_b.data[i * d_module + j] + b[j];
                fix_attn_output.data[i * d_module + j] &= ell_mask_;
            }
        }
        uint64_t *prime_fix_attn_output = new uint64_t[fix_attn_output.size];
        conv->Ring_to_Prime(fix_attn_output.data, prime_fix_attn_output, fix_attn_output.size, DEFAULT_ELL,
                            party->parm->plain_mod);
        BFVLongPlaintext fix_attn_output_plain(parm, prime_fix_attn_output, fix_attn_output.size);
        delete[] prime_fix_attn_output;
        rb_secret_b.multiply_plain_inplace(fix_attn_output_plain, party->parm->evaluator);
        return rb_secret_b; // alice hold it
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0, 1);
        double rb = dist(gen);
        uint64_t fix_rb = sci::neg_mod(static_cast<int64_t>(rb * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL));
        uint64_t fix_div_rb = sci::neg_mod(static_cast<int64_t>(1. / rb * (1ULL << (party->parm->plain_mod))),
                                           (1ULL << party->parm->plain_mod));
        FixArray fix_output_ =
            fpmath->fix->input(sci::BOB, output.size(), output.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_weight = fpmath->fix->input(sci::BOB, W.size(), W.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        fix_weight = fpmath->fix->mul(fix_weight, fix_rb);
        fix_weight = fpmath->fix->location_truncation(fix_weight, DEFAULT_SCALE);
        fix_weight.party = sci::PUBLIC;
        FixArray fix_attn_output = fpmath->dot(fix_output_, fix_weight, batch_size, d_module, d_module, DEFAULT_ELL);
        fix_attn_output = fpmath->fix->location_truncation(fix_attn_output, DEFAULT_SCALE);
        uint64_t ell_mask_ = fix_attn_output.ell_mask();
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < d_module; j++) {
                fix_attn_output.data[i * d_module + j] = fix_attn_output.data[i * d_module + j] + b[j];
                fix_attn_output.data[i * d_module + j] &= ell_mask_;
            }
        }
        fix_attn_output.party = sci::PUBLIC;
        fix_output_ = fpmath->fix->mul(fix_output_, fix_rb);
        fix_output_ = fpmath->fix->location_truncation(fix_output_, DEFAULT_SCALE);
        fix_output_.party = sci::PUBLIC;

        fpmath->fix->send_fix_array(fix_output_);
        fpmath->fix->send_fix_array(fix_weight);
        fpmath->fix->send_fix_array(fix_attn_output);
        BFVLongCiphertext rb_secret_b(parm, fix_div_rb, party);
        BFVLongCiphertext::send(io, &rb_secret_b);
    }
    // BFVLongCiphertext output_secret;
    // if (party->party == sci::ALICE) {
    //     BFVLongCiphertext::recv(io, &output_secret, party->parm->context);
    //     BFVLongPlaintext output_plain = BFVLongPlaintext(party->parm, output);
    //     output_secret.multiply_plain_inplace(output_plain, party->parm->evaluator);
    // } else {
    //     BFVLongCiphertext output_secret_b(BFVLongPlaintext(party->parm, output), party);
    //     BFVLongCiphertext::send(io, &output_secret_b);
    // }
#ifdef LOG
    STOP_TIMER("Multi-Head Attention")
    total_comm = io->counter - total_comm;
    printf("Multi-Head Attention Send data %ld Bytes. \n", total_comm);
#endif
    return BFVLongCiphertext();
}