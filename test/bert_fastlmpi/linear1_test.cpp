#include "linear1.h"
#include "model.h"
#include "protocols/fixed-protocol.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <cstdint>
#include <vector>

timestamp get_max_time(timestamp t[], int len) {
    timestamp max = t[0];
    for (int i = 1; i < len; i++) {
        if (max < t[i]) {
            max = t[i];
        }
    }
    return max;
}

timestamp get_timestamp() {
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    return time.count();
}

timestamp Encoder::forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv) {
    timestamp time = 0;
    vector<vector<uint64_t>> attn_output_h(n_heads);
    vector<uint64_t> tmp_output(batch_size * d_module);
    BFVLongCiphertext *inp_e = nullptr;

    /**
     *  START Multi-Head Attention
     */
    if (party->party == ALICE) {
        inp_e = RFCP_bfv_encodeA(input, party, batch_size, d_module, d_k);
        time += send_encoded_ciper(inp_e, fpmath, d_module);
    } else {
        inp_e = new BFVLongCiphertext[d_module];
        recv_encoded_ciper(inp_e, fpmath, d_module);
    }
    timestamp times[n_heads] = {0};
    // this can be multi thread
#pragma omp parallel for num_threads(12)
    for (int head = 0; head < n_heads; head++) {
        vector<uint64_t> Q, K, V;
        BFVLongCiphertext *Q_encode, *Q_encode_remote = new BFVLongCiphertext[d_k], *softmax_encode,
                                     *softmax_encode_remote = new BFVLongCiphertext[batch_size];
        timestamp start_QKV = get_timestamp();
        if (party->party == ALICE) {
            Q = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
            K = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
            V = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
        } else {
            BFVLongCiphertext Q_sec_a = RFCP_bfv_matmul(inp_e, nhead_WQ[head], batch_size, d_module, d_k, party->parm),
                              K_sec_a = RFCP_bfv_matmul(inp_e, nhead_WK[head], batch_size, d_module, d_k, party->parm),
                              V_sec_a = RFCP_bfv_matmul(inp_e, nhead_WV[head], batch_size, d_module, d_k, party->parm);
            BFVLongPlaintext bQ_plain(party->parm, nhead_bQ[head]), bK_plain(party->parm, nhead_bK[head]),
                bV_plain(party->parm, nhead_bV[head]);
            Q_sec_a.add_plain_inplace(bQ_plain, party->parm->evaluator);
            K_sec_a.add_plain_inplace(bK_plain, party->parm->evaluator);
            V_sec_a.add_plain_inplace(bV_plain, party->parm->evaluator);

            Q = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, Q_sec_a);
            K = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, K_sec_a);
            V = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, K_sec_a);
        }
        times[head] += (get_timestamp() - start_QKV);
        conv->Prime_to_Ring(party->party, V.data(), V.data(), batch_size * d_k, DEFAULT_ELL, party->parm->plain_mod,
                            DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        start_QKV = get_timestamp();
        vector<uint64_t> K_T(batch_size * d_k);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < d_k; j++) {
                K_T[j * batch_size + i] = K[i * d_k + j];
            }
        }
        vector<uint64_t> QK_local = matmul(Q, K, batch_size, d_k, batch_size, true);
        Q_encode = RFCP_bfv_encodeA(Q, party, batch_size, d_k, batch_size);
        times[head] += (get_timestamp() - start_QKV);
        if (party->party == ALICE) {
            timestamp start_send = get_timestamp();
            for (int i = 0; i < d_k; i++) {
                BFVLongCiphertext::send(fpmath[head]->iopack->io, Q_encode + i);
            }
            times[head] += (get_timestamp() - start_send);
            for (int i = 0; i < d_k; i++) {
                BFVLongCiphertext::recv(fpmath[head]->iopack->io, Q_encode_remote + i, party->parm->context);
            }
        } else {
            // recv_encoded_ciper(Q_encode_remote, fpmath, d_k);
            // times[head] += send_encoded_ciper(Q_encode, fpmath, d_k);
            for (int i = 0; i < d_k; i++) {
                BFVLongCiphertext::recv(fpmath[head]->iopack->io, Q_encode_remote + i, party->parm->context);
            }
            timestamp start_send = get_timestamp();
            for (int i = 0; i < d_k; i++) {
                BFVLongCiphertext::send(fpmath[head]->iopack->io, Q_encode + i);
            }
            times[head] += (get_timestamp() - start_send);
        }

        timestamp start_QK = get_timestamp();
        BFVLongCiphertext QK_enc = RFCP_bfv_matmul(Q_encode_remote, K_T, batch_size, d_k, batch_size, party->parm);
        vector<uint64_t> ret1, ret2;
        if (party->party == ALICE) {
            ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, QK_enc);
            ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
        } else {
            ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
            ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, QK_enc);
        }
        for (int i = 0; i < batch_size * batch_size; i++) {
            QK_local[i] = (QK_local[i] + ret1[i] + ret2[i]) & (1ULL << DEFAULT_ELL);
        }
        times[head] += get_timestamp() - start_QK;
        conv->Prime_to_Ring(party->party, QK_local.data(), QK_local.data(), batch_size * batch_size, DEFAULT_ELL,
                            party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        vector<uint64_t> softmax_output;
        // times[head] += softmax(QK_local, softmax_output, fpmath[head], conv);
        softmax(QK_local, softmax_output, fpmath[head], conv);

        timestamp start_SV_local = get_timestamp();
        attn_output_h[head] = matmul(softmax_output, V, batch_size, batch_size, d_k);
        softmax_encode = RFCP_bfv_encodeA(softmax_output, party, batch_size, batch_size, d_k);
        times[head] += get_timestamp() - start_SV_local;
        if (party->party == ALICE) {
            // recv_encoded_ciper(softmax_encode_remote, fpmath, batch_size);
            // times[head] += send_encoded_ciper(softmax_encode, fpmath, batch_size);
            timestamp start_send = get_timestamp();
            for (int i = 0; i < batch_size; i++) {
                BFVLongCiphertext::send(fpmath[head]->iopack->io, softmax_encode + i);
            }
            times[head] += (get_timestamp() - start_send);
            for (int i = 0; i < batch_size; i++) {
                BFVLongCiphertext::recv(fpmath[head]->iopack->io, softmax_encode_remote + i, party->parm->context);
            }
        } else {
            for (int i = 0; i < batch_size; i++) {
                BFVLongCiphertext::recv(fpmath[head]->iopack->io, softmax_encode_remote + i, party->parm->context);
            }
            timestamp start_send = get_timestamp();
            for (int i = 0; i < batch_size; i++) {
                BFVLongCiphertext::send(fpmath[head]->iopack->io, softmax_encode + i);
            }
            times[head] += (get_timestamp() - start_send);
        }

        timestamp start_SV = get_timestamp();
        BFVLongCiphertext attn_out_enc =
            RFCP_bfv_matmul(softmax_encode_remote, V, batch_size, batch_size, d_k, party->parm);
        vector<uint64_t> attn_out_ret1, attn_out_ret2;
        if (party->party == ALICE) {
            attn_out_ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, attn_out_enc);
            attn_out_ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
        } else {
            attn_out_ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
            attn_out_ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, attn_out_enc);
        }
        for (int i = 0; i < batch_size * d_k; i++) {
            attn_output_h[head][i] =
                (attn_output_h[head][i] + attn_out_ret1[i] + attn_out_ret2[i]) & (1ULL << DEFAULT_ELL);
        }
        times[head] += get_timestamp() - start_SV;

        delete[] softmax_encode_remote;
        delete[] softmax_encode;
        delete[] Q_encode_remote;
        delete[] Q_encode;
    }

    timestamp start_concat = get_timestamp();
#pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < d_k; j++) {
                tmp_output[i * d_module + h * d_k + j] = attn_output_h[h][i * d_k + j];
            }
        }
    }
    // vector<uint64_t> attn_out;
    // if (*party == BOB) {
    //     vector<uint64_t> attn_out_local = matmul(tmp_output, Attn_W, batch_size, d_module, d_module);
    //     for (int i = 0; i < batch_size * d_module; i++) {
    //         attn_out_local[i] = (attn_out_local[i] + Attn_b[i]) & (1ULL << DEFAULT_ELL);
    //     }
    //     BFVLongPlaintext attn_out_local_plain = BFVLongPlaintext(party->parm, attn_out_local);
    //     BFVLongCiphertext *tmpout_e = new BFVLongCiphertext[d_module];
    //     timestamp start_rev = get_timestamp();
    //     recv_encoded_ciper(tmpout_e, fpmath, d_module);
    //     time -= (get_timestamp() - start_rev);
    //     BFVLongCiphertext attn_out_enc = RFCP_bfv_matmul(tmpout_e, Attn_W, batch_size, d_module, d_module,
    //     party->parm); attn_out_enc.add_plain_inplace(attn_out_local_plain, party->parm->evaluator); attn_out =
    //     conv->he_to_ss_server(fpmath[layer]->iopack->io, party->parm, attn_out_enc);
    // } else {
    //     BFVLongCiphertext *tmpout_e = RFCP_bfv_encodeA(tmp_output, party, batch_size, d_module, d_module);
    //     send_encoded_ciper(tmpout_e, fpmath, d_module);
    //     attn_out = conv->he_to_ss_client(fpmath[layer]->iopack->io, party);
    // }
    timestamp end_concat = get_timestamp() - start_concat;
    time += end_concat;
    // time += get_max_time(times, n_heads);
    for (int i = 0; i < n_heads; i++) {
        time += times[i];
    }
    /**
     * END Multi-Head Attention
     */

    /**
     * START LayerNorm
     */

    /**
     * END LayerNorm
     */

    /**
     * START FFN
     */

    /**
     * END FFN
     */

    /**
     * START LayerNorm
     */

    /**
     * END LayerNorm
     */
    output = vector<uint64_t>(batch_size * d_module);
    delete[] inp_e;
    return time;
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
            iopack[i] = new IOPack(party_, 64789 + i, ip);
            if (party_ == BOB) {
                iopack[i]->io->is_server = true;
            }
            otpack[i] = new OTPack(iopack[i], party_);
            fpmath[i] = new FPMath(party_, iopack[i], otpack[i]);
        }

        Conversion *conv = new Conversion();
        BFVParm *parm =
            new BFVParm(8192, {40, 30, 30, 40},
                        default_prime_mod.at(29)); // new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, parm);

        vector<uint64_t> input(batch_size * d_module), output;
        random_ell_mat(input, DEFAULT_ELL);

        auto start = get_timestamp();
        Bert *bert = new Bert(party);
        auto end = get_timestamp() - start;
        std::cout << "time of load data: " << end << "\n";

        iopack[0]->io->sync();

        size_t start_comm = 0;
        for (int i = 0; i < N_THREADS; i++) {
            start_comm += fpmath[i]->iopack->get_comm();
        }
        std::cout << "time of forward: " << bert->encoders[0]->forward(input, output, fpmath, conv) << "\n";
        size_t end_comm = 0;
        for (int i = 0; i < N_THREADS; i++) {
            end_comm += fpmath[i]->iopack->get_comm();
        }
        std::cout << "comm: " << end_comm - start_comm << "\n";
        // std::cout << "time cost:" << time.count() / 1000000 << "\n";
        // 144 986 749 1026
        delete bert;
        delete party;
        delete parm;
        delete conv;
        for (int i = 0; i < N_THREADS; i++) {
            delete fpmath[i];
            delete otpack[i];
            delete iopack[i];
        }
    } else {
        std::cout << "No party input\n";
    }
}