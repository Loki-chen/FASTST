#include "FixedPoint/fixed-math.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <ctime>
#include <model.h>
#include <utils.h>
#define DEFAULT_SCALE 12
#define DEFAULT_ELL 37
#define N_THREADS 12
typedef unsigned long long timestamp;

timestamp get_timestamp();

using namespace sci;

const string base_path = "/data/BOLT/bolt/prune/mrpc/weights_txt/";
const string WQ_path = "bert.encoder.layer.LAYER.attention.self.query.weight.txt",
             WK_path = "bert.encoder.layer.LAYER.attention.self.key.weight.txt",
             WV_path = "bert.encoder.layer.LAYER.attention.self.value.weight.txt",
             bQ_path = "bert.encoder.layer.LAYER.attention.self.query.bias.txt",
             bK_path = "bert.encoder.layer.LAYER.attention.self.key.bias.txt",
             bV_path = "bert.encoder.layer.LAYER.attention.self.value.bias.txt",
             Attn_W_path = "bert.encoder.layer.LAYER.attention.output.dense.weight.txt",
             Attn_b_path = "bert.encoder.layer.LAYER.attention.output.dense.bias.txt",
             gamma1_path = "bert.encoder.layer.LAYER.attention.output.LayerNorm.weight.txt",
             beta1_path = "bert.encoder.layer.LAYER.attention.output.LayerNorm.bias.txt",
             W1_path = "bert.encoder.layer.LAYER.intermediate.dense.weight.txt",
             b1_path = "bert.encoder.layer.LAYER.intermediate.dense.bias.txt",
             W2_path = "bert.encoder.layer.LAYER.output.dense.weight.txt",
             b2_path = "bert.encoder.layer.LAYER.output.dense.bias.txt",
             gamma2_path = "bert.encoder.layer.LAYER.output.LayerNorm.weight.txt",
             beta2_path = "bert.encoder.layer.LAYER.output.LayerNorm.bias.txt";

class Encoder {
    timestamp send_encoded_ciper(BFVLongCiphertext *inp_e, FPMath **fpmath, int length) {
        thread send_threads[N_THREADS];
        int split = length / N_THREADS;
        if (split * N_THREADS > length) {
            split += 1;
        }
        for (int t = 0; t < N_THREADS; t++) {
            int num_ops = split;
            if (t == N_THREADS - 1) {
                num_ops = length - split * (N_THREADS - 1);
            }
            send_threads[t] = std::thread([t, num_ops, split, fpmath, inp_e]() {
                for (int i = 0; i < num_ops; i++) {
                    BFVLongCiphertext::send(fpmath[t]->iopack->io, inp_e + t * split + i);
                }
            });
        }
        timestamp start = get_timestamp();
        for (int t = 0; t < N_THREADS; t++) {
            send_threads[t].join();
        }
        timestamp end = get_timestamp() - start;
        return end;
    }

    void recv_encoded_ciper(BFVLongCiphertext *inp_e, FPMath **fpmath, int length) {
        thread recv_threads[N_THREADS];
        int split = length / N_THREADS;
        if (split * N_THREADS > length) {
            split += 1;
        }
        for (int t = 0; t < N_THREADS; t++) {
            int num_ops = split;
            if (t == N_THREADS - 1) {
                num_ops = length - split * (N_THREADS - 1);
            }
            recv_threads[t] = std::thread([t, num_ops, split, fpmath, inp_e, this]() {
                for (int i = 0; i < num_ops; i++) {
                    BFVLongCiphertext::recv(fpmath[t]->iopack->io, inp_e + t * split + i, party->parm->context);
                }
            });
        }
        for (int t = 0; t < N_THREADS; t++) {
            recv_threads[t].join();
        }
    }

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

    timestamp softmax(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath *fpmath, Conversion *conv) {
        timestamp time;
        size_t size = batch_size * batch_size;
        FixArray fix_inp = fpmath->fix->input(PUBLIC, size, input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray exp_inp = fpmath->location_exp(fix_inp, DEFAULT_SCALE, DEFAULT_SCALE);
        if (party->party == ALICE) {
            BFVLongCiphertext exp_sec_b = conv->ss_to_he_server(party->parm, fpmath->iopack->io, exp_inp.data, exp_inp.size, false);
            vector<uint64_t> R(size);
            // random_modP_mat(R, party->parm->plain_mod);
            BFVLongPlaintext R_plain(party->parm, R);
            auto exp_R_sec_b = exp_sec_b.add_plain(R_plain, party->parm->evaluator);
            vector<FixArray> fix_R(batch_size);
#pragma omp parallel for
            for (int i = 0; i < batch_size; i++) {
                fix_R[i] =
                    fpmath->fix->input(party->party, batch_size, R.data() + i * batch_size, true, DEFAULT_ELL, DEFAULT_SCALE);
            }
            FixArray fix_SR = fpmath->fix->tree_sum(fix_R);
            BFVLongPlaintext SR_plain(party->parm, fix_SR.data, fix_SR.size);
            BFVLongCiphertext SR_sec_a(SR_plain, party);
            BFVLongCiphertext::send(fpmath->iopack->io, &exp_R_sec_b);
            BFVLongCiphertext::send(fpmath->iopack->io, &SR_sec_a);

            BFVLongCiphertext S_exp_V, V_sec_b;
            BFVLongCiphertext::recv(fpmath->iopack->io, &S_exp_V, party->parm->context);
            BFVLongCiphertext::recv(fpmath->iopack->io, &V_sec_b, party->parm->context);
            BFVLongPlaintext S_exp_V_plain = S_exp_V.decrypt(party);
            vector<uint64_t> Sexp_V = S_exp_V_plain.decode_uint(party->parm), Sexp_V_expand(batch_size * batch_size);
#pragma omp parallel for
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < batch_size; j++) {
                    Sexp_V_expand[i * batch_size + j] = mod_inverse(Sexp_V[i], party->parm->plain_mod);
                }
            }

            // std::cout << "1 / (EV): "<< modInverse(Sexp_V[0], party->parm->plain_mod) << " " << modInverse(Sexp_V[1],
            // party->parm->plain_mod) << "\n"; // 1 / EV
            BFVLongPlaintext Sexp_expand_plain(party->parm, Sexp_V_expand);
            V_sec_b.multiply_plain_inplace(Sexp_expand_plain, party->parm->evaluator);
            exp_sec_b.multiply_inplace(V_sec_b, party->parm->evaluator);

            BFVLongCiphertext::send(fpmath->iopack->io, &exp_sec_b);
            output = conv->he_to_ss_server(fpmath->iopack->io, party->parm, exp_sec_b);
        } else {
            conv->ss_to_he_client(party, fpmath->iopack->io, exp_inp.data, exp_inp.size, DEFAULT_ELL);

            BFVLongCiphertext exp_sec_b, R_sec_a, SR_sec_a;
            BFVLongCiphertext::recv(fpmath->iopack->io, &exp_sec_b, party->parm->context);
            BFVLongCiphertext::recv(fpmath->iopack->io, &SR_sec_a, party->parm->context);
            BFVLongPlaintext exp_R_plain = exp_sec_b.decrypt(party);
            vector<uint64_t> exp_R = exp_R_plain.decode_uint(party->parm);
            vector<FixArray> fix_exp_R(batch_size);
#pragma omp parallel for
            for (int i = 0; i < batch_size; i++) {
                fix_exp_R[i] =
                    fpmath->fix->input(party->party, batch_size, exp_R.data() + i * batch_size, true, DEFAULT_ELL, DEFAULT_SCALE);
            }
            FixArray fix_S_exp_R = fpmath->fix->tree_sum(fix_exp_R);
            // std::cout << "1/E: " << modInverse(fix_S_exp_R.data[0], party->parm->plain_mod) << " " <<
            // modInverse(fix_S_exp_R.data[1], party->parm->plain_mod)<< "\n";
            BFVLongPlaintext S_exp_R_plain(party->parm, fix_S_exp_R.data, fix_S_exp_R.size);
            SR_sec_a.negate_inplace(party->parm->evaluator);
            SR_sec_a.add_plain_inplace(S_exp_R_plain, party->parm->evaluator);
            vector<uint64_t> V(batch_size, 1), V_expand(batch_size * batch_size);
            // random_modP_mat(V, party->parm->plain_mod);
#pragma omp parallel for
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < batch_size; j++) {
                    V_expand[i * batch_size + j] = V[i];
                }
            }
            // std::cout << "V: " << V_expand[0] << " " << V_expand[3] << "\n";
            BFVLongPlaintext V_plain(party->parm, V), V_expand_plain(party->parm, V_expand);
            SR_sec_a.multiply_plain_inplace(V_plain, party->parm->evaluator);
            BFVLongCiphertext V_sec_b(V_expand_plain, party);
            BFVLongCiphertext::send(fpmath->iopack->io, &SR_sec_a);
            BFVLongCiphertext::send(fpmath->iopack->io, &V_sec_b);

            output = conv->he_to_ss_client(fpmath->iopack->io, party);
        }
        return time;
    }

    bfv_matrix matmul(const bfv_matrix &mat1, const bfv_matrix &mat2, size_t dim1, size_t dim2, size_t dim3,
                      bool trans = false) {
        bfv_matrix result(dim1 * dim3);
        uint64_t ret_mask = 1ULL << DEFAULT_ELL;
        if (!trans) {
            {
#pragma omp parallel for
                for (size_t i = 0; i < dim1; i++) {
                    const size_t base_idx1 = i * dim2;
                    const size_t base_idx2 = i * dim3;
                    for (size_t k = 0; k < dim2; k++) {
                        const size_t base_idx3 = k * dim3;
                        const uint64_t tmp = mat1[base_idx1 + k];
                        for (size_t j = 0; j < dim3; j++) {
                            result[base_idx2 + j] += tmp * mat2[base_idx3 + j];
                            result[base_idx2 + j] &= ret_mask;
                        }
                    }
                }
            }
        } else {
            {
#pragma omp parallel for
                for (size_t i = 0; i < dim1; i++) {
                    const size_t base_idx1 = i * dim2;
                    const size_t base_idx2 = i * dim3;
                    for (size_t j = 0; j < dim3; j++) {
                        const size_t base_idx3 = j * dim2;
                        uint64_t sum = 0.;
                        for (size_t k = 0; k < dim2; k++) {
                            sum += mat1[base_idx1 + k] * mat2[base_idx3 + k];
                        }
                        result[base_idx2 + j] = (sum & ret_mask);
                    }
                }
            }
        }
        for (size_t i = 0; i < dim1 * dim3; i++) {
            result[i] = (result[i] >> DEFAULT_SCALE) & ret_mask;
        }
        return result;
    }

public:
    BFVKey *party;
    int layer;
    vector<uint64_t> WQ, WK, WV, bQ, bK, bV, Attn_W, Attn_b, gamma1, beta1, W1, b1, W2, b2, gamma2, beta2, null_vector;
    vector<vector<uint64_t>> nhead_WQ, nhead_WK, nhead_WV, nhead_bQ, nhead_bK, nhead_bV;
    BFVLongCiphertext null_ciper;
    timestamp time;

    Encoder(BFVKey *_party, int _layer) : party(_party), layer(_layer) {
        if (party->party == BOB) {
            string layer_str = std::to_string(layer);
            load_bfv_mat(WQ, replace(base_path + WQ_path, "LAYER", layer_str));
            load_bfv_mat(WK, replace(base_path + WK_path, "LAYER", layer_str));
            load_bfv_mat(WV, replace(base_path + WV_path, "LAYER", layer_str));
            load_bfv_mat(bQ, replace(base_path + bQ_path, "LAYER", layer_str));
            load_bfv_mat(bK, replace(base_path + bK_path, "LAYER", layer_str));
            load_bfv_mat(bV, replace(base_path + bV_path, "LAYER", layer_str));
            load_bfv_mat(Attn_W, replace(base_path + Attn_W_path, "LAYER", layer_str));
            load_bfv_mat(Attn_b, replace(base_path + Attn_b_path, "LAYER", layer_str));
            load_bfv_mat(gamma1, replace(base_path + gamma1_path, "LAYER", layer_str));
            load_bfv_mat(beta1, replace(base_path + beta1_path, "LAYER", layer_str));
            load_bfv_mat(W1, replace(base_path + W1_path, "LAYER", layer_str));
            load_bfv_mat(b1, replace(base_path + b1_path, "LAYER", layer_str));
            load_bfv_mat(W2, replace(base_path + W2_path, "LAYER", layer_str));
            load_bfv_mat(b2, replace(base_path + b2_path, "LAYER", layer_str));
            load_bfv_mat(gamma2, replace(base_path + gamma1_path, "LAYER", layer_str));
            load_bfv_mat(beta2, replace(base_path + beta1_path, "LAYER", layer_str));

            nhead_WQ.resize(n_heads);
            nhead_WK.resize(n_heads);
            nhead_WV.resize(n_heads);
            nhead_bQ.resize(n_heads);
            nhead_bK.resize(n_heads);
            nhead_bV.resize(n_heads);

            size_t size = batch_size * d_k;
#pragma omp parallel for
            for (int i = 0; i < n_heads; i++) {
                nhead_WQ[i] = bfv_matrix(WQ.begin() + i * size, WQ.begin() + (i + 1) * size);
                nhead_WK[i] = bfv_matrix(WK.begin() + i * size, WK.begin() + (i + 1) * size);
                nhead_WV[i] = bfv_matrix(WV.begin() + i * size, WV.begin() + (i + 1) * size);
                vector<uint64_t> tmp_bQ = bfv_matrix(bQ.begin() + i * d_k, bQ.begin() + (i + 1) * d_k),
                                 tmp_bK = bfv_matrix(bK.begin() + i * d_k, bK.begin() + (i + 1) * d_k),
                                 tmp_bV = bfv_matrix(bV.begin() + i * d_k, bV.begin() + (i + 1) * d_k);
                nhead_bQ[i].resize(batch_size * d_k);
                nhead_bK[i].resize(batch_size * d_k);
                nhead_bV[i].resize(batch_size * d_k);
                for (size_t k = 0; k < batch_size; k++) {
                    for (size_t j = 0; j < d_k; j++) {
                        nhead_bQ[i][k * d_k + j] = tmp_bQ[j];
                        nhead_bK[i][k * d_k + j] = tmp_bK[j];
                        nhead_bV[i][k * d_k + j] = tmp_bV[j];
                    }
                }
            }
        }
    }

    timestamp forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv);
};

class Bert {
public:
    BFVKey *party;
    vector<Encoder *> encoders;
    timestamp time;

    Bert(BFVKey *_party) : party(_party) {
        encoders.resize(n_layer);
        for (int i = 0; i < n_layer; i++) {
            encoders[i] = new Encoder(party, i);
        }
    }

    ~Bert() {
        for (int i = 0; i < n_layer; i++) {
            delete encoders[i];
        }
    }

    timestamp forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv);
};