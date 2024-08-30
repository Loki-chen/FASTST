#include "bert.h"
#include "model.h"
#include "utils/he-bfv.h"

timestamp get_timestamp() {
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    return time.count();
}

timestamp Encoder::forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv) {
    timestamp time = 0;
    vector<vector<uint64_t>> attn_output_h(n_heads);
    vector<uint64_t> tmp_output(batch_size * d_module);
    int size = batch_size * d_module;
    BFVLongCiphertext *inp_e = nullptr;

    if (party->party == ALICE) {
        inp_e = RFCP_bfv_encodeA(input, party, batch_size, d_module, d_k);
        time += send_encoded_ciper(inp_e, fpmath, d_module);
    } else {
        inp_e = new BFVLongCiphertext[d_module];
        recv_encoded_ciper(inp_e, fpmath, d_module);
    }
    timestamp times[n_heads];

    // this can be multi thread
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
            times[head] += send_encoded_ciper(Q_encode, fpmath, d_k);
            recv_encoded_ciper(Q_encode_remote, fpmath, d_k);
        } else {
            recv_encoded_ciper(Q_encode_remote, fpmath, d_k);
            times[head] += send_encoded_ciper(Q_encode, fpmath, d_k);
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
        conv->Prime_to_Ring(party->party, ret1.data(), ret1.data(), batch_size * batch_size, DEFAULT_ELL,
                            party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        conv->Prime_to_Ring(party->party, ret2.data(), ret2.data(), batch_size * batch_size, DEFAULT_ELL,
                            party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        for (int i = 0; i < batch_size * batch_size; i++) {
            QK_local[i] = (QK_local[i] + ret1[i] + ret2[i]) & (1ULL << DEFAULT_ELL);
        }
        vector<uint64_t> softmax_output;
        times[head] += get_timestamp() - start_QK;
        times[head] += softmax(QK_local, softmax_output, fpmath[head], conv);

        timestamp start_SV_local = get_timestamp();
        attn_output_h[head] = matmul(softmax_output, V, batch_size, batch_size, d_k);
        softmax_encode = RFCP_bfv_encodeA(softmax_output, party, batch_size, batch_size, d_k);
        times[head] += get_timestamp() - start_SV_local;
        if (party->party == ALICE) {
            times[head] += send_encoded_ciper(softmax_encode, fpmath, batch_size);
            // recv_encoded_ciper(softmax_encode_remote, fpmath, batch_size);
        } else {
            // recv_encoded_ciper(softmax_encode_remote, fpmath, batch_size);
            // times[head] += send_encoded_ciper(softmax_encode, fpmath, batch_size);
        }

        // timestamp start_SV = get_timestamp();
        // BFVLongCiphertext attn_out_enc =
        //     RFCP_bfv_matmul(softmax_encode_remote, V, batch_size, batch_size, d_k, party->parm);
        // vector<uint64_t> attn_out_ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, QK_enc),
        //                  attn_out_ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
        // conv->Prime_to_Ring(party->party, attn_out_ret1.data(), attn_out_ret1.data(), batch_size * d_k,
        //                     DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        // conv->Prime_to_Ring(party->party, attn_out_ret2.data(), attn_out_ret2.data(), batch_size * d_k,
        //                     DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
        // for (int i = 0; i < batch_size * batch_size; i++) {
        //     attn_output_h[head][i] =
        //         (attn_output_h[head][i] + attn_out_ret1[i] + attn_out_ret2[i]) & (1ULL << DEFAULT_ELL);
        // }
        // times[head] += get_timestamp() - start_SV;

        delete[] softmax_encode_remote;
        delete[] softmax_encode;
        delete[] Q_encode_remote;
        delete[] Q_encode;
    }

    timestamp start = get_timestamp();
#pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < d_k; j++) {
                tmp_output[i * d_module + h * d_k + j] = attn_output_h[h][i * d_k + j];
            }
        }
    }
    timestamp end = get_timestamp() - start;
    time += end;
    output = vector<uint64_t>(batch_size * d_module);
    delete[] inp_e;
    return time;
}

timestamp Bert::forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv) {
    vector<uint64_t> upper_out;
    timestamp time = encoders[0]->forward(input, upper_out, fpmath, conv);
    for (int i = 1; i < n_layer; i++) {
        time += encoders[i]->forward(upper_out, output, fpmath, conv);
        upper_out = output;
    }
    return time;
}