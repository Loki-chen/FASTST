#include "bert.h"
#include "FixedPoint/fixed-math.h"
#include "linear.h"
#include "protocols/fixed-protocol.h"
#include "utils/conversion.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include <cstdint>
#include <ctime>
#include <model.h>
#include <seal/context.h>
#include <utils.h>
#define N_THREADS 12

#ifdef BERT_PERF
double t_total_linear1 = 0;
double t_total_linear2 = 0;
double t_total_linear3 = 0;
double t_total_linear4 = 0;

double t_total_pruning = 0;

double t_total_softmax = 0;
double t_total_mul_v = 0;
double t_total_gelu = 0;
double t_total_ln_1 = 0;
double t_total_ln_2 = 0;
double t_total_tanh = 0;

double t_total_repacking = 0;
double t_total_gt_sub = 0;
double t_total_shift = 0;

double t_total_conversion = 0;

double t_total_ln_share = 0;

uint64_t c_linear_1 = 0;
uint64_t c_linear_2 = 0;
uint64_t c_linear_3 = 0;
uint64_t c_linear_4 = 0;
uint64_t c_softmax = 0;
uint64_t c_pruning = 0;
uint64_t c_gelu = 0;
uint64_t c_ln1 = 0;
uint64_t c_ln2 = 0;
uint64_t c_softmax_v = 0;
uint64_t c_shift = 0;
uint64_t c_gt_sub = 0;
uint64_t c_tanh = 0;
uint64_t c_pc = 0;

uint64_t r_linear_1 = 0;
uint64_t r_linear_2 = 0;
uint64_t r_linear_3 = 0;
uint64_t r_linear_4 = 0;
uint64_t r_softmax = 0;
uint64_t r_pruning = 0;
uint64_t r_gelu = 0;
uint64_t r_ln1 = 0;
uint64_t r_ln2 = 0;
uint64_t r_softmax_v = 0;
uint64_t r_shift = 0;
uint64_t r_gt_sub = 0;
uint64_t r_tanh = 0;
uint64_t r_pc = 0;

double n_rounds = 0;
double n_comm = 0;

void send_encrypted_vector(sci::NetIO *io, vector<Ciphertext> &cipher) {
    BFVLongCiphertext lct;
    lct.cipher_data = cipher;
    lct.len = cipher.size() * 8192;
    BFVLongCiphertext::send(io, &lct);
}

void recv_encrypted_vector(SEALContext *context, sci::NetIO *io, vector<Ciphertext> &cipher) {
    BFVLongCiphertext lct;
    BFVLongCiphertext::recv(io, &lct, context);
    cipher = lct.cipher_data;
}

void send_encoded_ciper(BFVLongCiphertext *inp_e, FPMath **fpmath, int length) {
    thread send_threads[N_THREADS];
    int split = length / N_THREADS;
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
    for (int t = 0; t < N_THREADS; t++) {
        send_threads[t].join();
    }
}

void recv_encoded_ciper(BFVLongCiphertext *inp_e, FPMath **fpmath, int length, SEALContext *context) {
    thread recv_threads[N_THREADS];
    int split = length / N_THREADS;
    for (int t = 0; t < N_THREADS; t++) {
        int num_ops = split;
        if (t == N_THREADS - 1) {
            num_ops = length - split * (N_THREADS - 1);
        }
        recv_threads[t] = std::thread([t, num_ops, split, fpmath, inp_e, context]() {
            for (int i = 0; i < num_ops; i++) {
                BFVLongCiphertext::recv(fpmath[t]->iopack->io, inp_e + t * split + i, context);
            }
        });
    }
    for (int t = 0; t < N_THREADS; t++) {
        recv_threads[t].join();
    }
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

inline uint64_t Bert::get_comm() {
    uint64_t total_comm = io->counter;
    for (int i = 0; i < MAX_THREADS; i++) {
        total_comm += nl.iopackArr[i]->get_comm();
    }
    uint64_t ret_comm = total_comm - n_comm;
    n_comm = total_comm;
    return ret_comm;
}

inline uint64_t Bert::get_round() {
    uint64_t total_round = io->num_rounds;
    for (int i = 0; i < MAX_THREADS; i++) {
        total_round += nl.iopackArr[i]->get_rounds();
    }
    uint64_t ret_round = total_round - n_rounds;
    n_rounds = total_round;
    return ret_round;
}
#endif

inline double interval(chrono::_V2::system_clock::time_point start) {
    auto end = high_resolution_clock::now();
    auto interval = (end - start) / 1e+9;
    return interval.count();
}

string replace_2(string str, string substr1, string substr2) {
    size_t index = str.find(substr1, 0);
    str.replace(index, substr1.length(), substr2);
    return str;
}

void save_to_file(uint64_t *matrix, size_t rows, size_t cols, const char *filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << (int64_t)matrix[i * cols + j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void save_to_file_vec(vector<vector<uint64_t>> matrix, size_t rows, size_t cols, const char *filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << matrix[i][j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void print_pt(HE *he, Plaintext &pt, int len) {
    vector<uint64_t> dest(len, 0ULL);
    he->encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for (int i = 0; i < 16; i++) {
        if (dest[i] > he->plain_mod_2) {
            cout << (int64_t)(dest[i] - he->plain_mod) << " ";
        } else {
            cout << dest[i] << " ";
        }
    }
    cout << endl;
}

void print_ct(HE *he, Ciphertext &ct, int len) {
    Plaintext pt;
    he->decryptor->decrypt(ct, pt);
    cout << "Noise budget: ";
    cout << YELLOW << he->decryptor->invariant_noise_budget(ct) << " ";
    cout << RESET << endl;
    print_pt(he, pt, len);
}

Bert::Bert(int party, int port, string address, string model_path, bool prune) {
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io, prune);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port + 1);

    this->prune = prune;

    if (party == ALICE) {
        cout << "> Loading and preprocessing weights on server" << endl;
#ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
#endif

        struct BertModel bm = load_model(model_path, NUM_CLASS);

#ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
#endif

        lin.weights_preprocess(bm);

#ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
#endif
    }
    cout << "> Bert intialized done!" << endl << endl;
}

Bert::~Bert() {}

void Bert::he_to_ss_server(HE *he, vector<Ciphertext> in, uint64_t *output, bool ring) {
#ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
#endif

    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
    prg.random_mod_p<uint64_t>(output, dim * slot_count, he->plain_mod);

    Plaintext pt_p_2;
    vector<uint64_t> p_2(slot_count, he->plain_mod_2);
    he->encoder->encode(p_2, pt_p_2);

    vector<Ciphertext> cts;
    for (int i = 0; i < dim; i++) {
        vector<uint64_t> tmp(slot_count);
        for (int j = 0; j < slot_count; ++j) {
            tmp[j] = output[i * slot_count + j];
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct;
        he->evaluator->sub_plain(in[i], pt, ct);
        if (ring) {
            he->evaluator->add_plain_inplace(ct, pt_p_2);
        }
        // print_pt(he, pt, 8192);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
#ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
#endif
}

vector<Ciphertext> Bert::ss_to_he_server(HE *he, uint64_t *input, int length, int bw) {
#ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
#endif
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Plaintext> share_server;
    int dim = length / slot_count;
    for (int i = 0; i < dim; i++) {
        vector<uint64_t> tmp(slot_count);
        for (int j = 0; j < slot_count; ++j) {
            tmp[j] = neg_mod(signed_val(input[i * slot_count + j], bw), (int64_t)plain_mod);
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        share_server.push_back(pt);
    }

    vector<Ciphertext> share_client(dim);
    recv_encrypted_vector(he->context, io, share_client);
    for (int i = 0; i < dim; i++) {
        he->evaluator->add_plain_inplace(share_client[i], share_server[i]);
    }
#ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
#endif
    return share_client;
}

void Bert::he_to_ss_client(HE *he, uint64_t *output, int length, const FCMetadata &data) {
#ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
#endif
    vector<Ciphertext> cts(length);
    recv_encrypted_vector(he->context, io, cts);
    for (int i = 0; i < length; i++) {
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext tmp;
        he->decryptor->decrypt(cts[i], tmp);
        he->encoder->decode(tmp, plain);
        std::copy(plain.begin(), plain.end(), &output[i * data.slot_count]);
    }
#ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
#endif
}

void Bert::ss_to_he_client(HE *he, uint64_t *input, int length, int bw) {
#ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
#endif
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Ciphertext> cts;
    int dim = length / slot_count;
    for (int i = 0; i < dim; i++) {
        vector<uint64_t> tmp(slot_count);
        for (int j = 0; j < slot_count; ++j) {
            tmp[j] = neg_mod(signed_val(input[i * slot_count + j], bw), (int64_t)plain_mod);
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct;
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
#ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
#endif
}

void Bert::ln_share_server(int layer_id, vector<uint64_t> &wln_input, vector<uint64_t> &bln_input, uint64_t *wln,
                           uint64_t *bln, const FCMetadata &data) {
#ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
#endif

    int length = 2 * COMMON_DIM;
    uint64_t *random_share = new uint64_t[length];

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for (int i = 0; i < length; i++) {
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length * sizeof(uint64_t));

    for (int i = 0; i < COMMON_DIM; i++) {
        random_share[i] = (wln_input[i] - random_share[i]) & mask_x;
        random_share[i + COMMON_DIM] = (bln_input[i] - random_share[i + COMMON_DIM]) & mask_x;
    }

    for (int i = 0; i < data.image_size; i++) {
        memcpy(&wln[i * COMMON_DIM], random_share, COMMON_DIM * sizeof(uint64_t));
        memcpy(&bln[i * COMMON_DIM], &random_share[COMMON_DIM], COMMON_DIM * sizeof(uint64_t));
    }

    delete[] random_share;
#ifdef BERT_PERF
    t_total_ln_share += interval(t_ln_share);
#endif
}

void Bert::ln_share_client(uint64_t *wln, uint64_t *bln, const FCMetadata &data) {

#ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
#endif

    int length = 2 * COMMON_DIM;

    uint64_t *share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    for (int i = 0; i < data.image_size; i++) {
        memcpy(&wln[i * COMMON_DIM], share, COMMON_DIM * sizeof(uint64_t));
        memcpy(&bln[i * COMMON_DIM], &share[COMMON_DIM], COMMON_DIM * sizeof(uint64_t));
    }
    delete[] share;
#ifdef BERT_PERF
    t_total_ln_share += interval(t_ln_share);
#endif
}

void Bert::pc_bw_share_server(uint64_t *wp, uint64_t *bp, uint64_t *wc, uint64_t *bc) {
    int wp_len = COMMON_DIM * COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM * NUM_CLASS;
    int bc_len = NUM_CLASS;

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    int length = wp_len + bp_len + wc_len + bc_len;
    uint64_t *random_share = new uint64_t[length];

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for (int i = 0; i < length; i++) {
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length * sizeof(uint64_t));

    // Write wp share
    int offset = 0;
    for (int i = 0; i < COMMON_DIM; i++) {
        for (int j = 0; j < COMMON_DIM; j++) {
            wp[i * COMMON_DIM + j] = (lin.w_p[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

    // Write bp share
    for (int i = 0; i < COMMON_DIM; i++) {
        bp[i] = (lin.b_p[i] - random_share[offset]) & mask_x;
        offset++;
    }

    // Write w_c share
    for (int i = 0; i < COMMON_DIM; i++) {
        for (int j = 0; j < NUM_CLASS; j++) {
            wc[i * NUM_CLASS + j] = (lin.w_c[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

    // Write b_c share
    for (int i = 0; i < NUM_CLASS; i++) {
        bc[i] = (lin.b_c[i] - random_share[offset]) & mask_x;
        offset++;
    }
}

void Bert::pc_bw_share_client(uint64_t *wp, uint64_t *bp, uint64_t *wc, uint64_t *bc) {
    int wp_len = COMMON_DIM * COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM * NUM_CLASS;
    int bc_len = NUM_CLASS;
    int length = wp_len + bp_len + wc_len + bc_len;

    uint64_t *share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    memcpy(wp, share, wp_len * sizeof(uint64_t));
    memcpy(bp, &share[wp_len], bp_len * sizeof(uint64_t));
    memcpy(wc, &share[wp_len + bp_len], wc_len * sizeof(uint64_t));
    memcpy(bc, &share[wp_len + bp_len + wc_len], bc_len * sizeof(uint64_t));
}

void Bert::softmax_v(HE *he, vector<Ciphertext> enc_v, uint64_t *s_softmax, uint64_t *s_v, uint64_t *s_softmax_v,
                     const FCMetadata &data) {
    if (party == ALICE) {
        // Server

        vector<vector<vector<Plaintext>>> S2_pack = lin.preprocess_softmax_s2(he, s_softmax, data);

        vector<Ciphertext> enc_s1_pack(data.image_size * data.image_size * 12 / data.slot_count);

        recv_encrypted_vector(he->context, io, enc_s1_pack);

        vector<vector<vector<Plaintext>>> R_pack = lin.preprocess_softmax_v_r(he, s_v, data);

        vector<Ciphertext> softmax_V_result(12 * data.image_size * data.filter_w / data.slot_count);

        lin.bert_softmax_V(he, enc_s1_pack, S2_pack, enc_v, R_pack, data, softmax_V_result);

        // send_encrypted_vector(io, softmax_V_result);

        uint64_t *softmax_v_server = new uint64_t[12 * data.image_size * data.filter_w];

        he_to_ss_server(he, softmax_V_result, softmax_v_server, true);

        lin.plain_cross_packing_postprocess_v(softmax_v_server, s_softmax_v, true, data);

        // for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
        //     s_softmax_v[i] = 0;
        // }

        delete[] softmax_v_server;

    } else {
        // Client

        uint64_t *softmax_v_client = new uint64_t[12 * data.image_size * data.filter_w];
        uint64_t *softmax_v_server = new uint64_t[12 * data.image_size * data.filter_w];

        vector<Ciphertext> enc_s1 = lin.preprocess_softmax_s1(he, s_softmax, data);

        send_encrypted_vector(io, enc_s1);

        // TODO: s_v column packing
        lin.client_S1_V_R(he, s_softmax, s_v, softmax_v_client, data);

        int cts_len = 12 * data.image_size * data.filter_w / data.slot_count;
        he_to_ss_client(he, softmax_v_server, cts_len, data);

        lin.plain_cross_packing_postprocess_v(softmax_v_server, s_softmax_v, true, data);

        for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
            if (softmax_v_client[i] > he->plain_mod_2) {
                s_softmax_v[i] += softmax_v_client[i] - he->plain_mod;
            } else {
                s_softmax_v[i] += softmax_v_client[i];
            }
            s_softmax_v[i] = neg_mod((int64_t)s_softmax_v[i], (int64_t)he->plain_mod);
        }

        delete[] softmax_v_client;
        delete[] softmax_v_server;
    }
}

void Bert::print_p_share(uint64_t *s, uint64_t p, int len) {
    if (party == ALICE) {
        io->send_data(s, len * sizeof(uint64_t));
    } else {
        uint64_t *s1 = new uint64_t[len];
        io->recv_data(s1, len * sizeof(uint64_t));

        for (int i = 0; i < len; i++) {
            uint64_t tmp = (s[i] + s1[i]) % p;
            if (tmp > (p / 2)) {
                tmp -= p;
            }
            cout << (int64_t)tmp << " ";
        }
        cout << endl;
    }
}

void Bert::check_p_share(uint64_t *s, uint64_t p, int len, uint64_t *ref) {
    if (party == ALICE) {
        io->send_data(s, len * sizeof(uint64_t));
    } else {
        uint64_t *s1 = new uint64_t[len];
        io->recv_data(s1, len * sizeof(uint64_t));

        for (int i = 0; i < len; i++) {
            uint64_t tmp = (s[i] + s1[i]) % p;
            if (tmp > (p / 2)) {
                tmp -= p;
            }
            if ((int64_t)tmp != (int64_t)ref[i]) {
                cout << "Error: " << (int64_t)tmp << " " << (int64_t)ref[i] << endl;
            }
        }
    }
}

vector<double> Bert::run(string input_fname, string mask_fname) {
    // Server: Alice
    // Client: Bob
    vector<uint64_t> softmax_mask;
    uint64_t h1_cache_12_original[INPUT_DIM * COMMON_DIM] = {0};
    uint64_t h1_cache_12[INPUT_DIM * COMMON_DIM] = {0};
    uint64_t h4_cache_12[INPUT_DIM * COMMON_DIM] = {0};
    uint64_t h98[COMMON_DIM] = {0};

    vector<Ciphertext> h1;
    vector<Ciphertext> h2;
    vector<Ciphertext> h4;
    vector<Ciphertext> h6;

#ifdef BERT_PERF
    n_rounds += io->num_rounds;
    n_comm += io->counter;

    for (int i = 0; i < MAX_THREADS; i++) {
        n_rounds += nl.iopackArr[i]->get_rounds();
        n_comm += nl.iopackArr[i]->get_comm();
    }

    auto t_linear1 = high_resolution_clock::now();
    auto t_linear2 = high_resolution_clock::now();
    auto t_linear3 = high_resolution_clock::now();
    auto t_linear4 = high_resolution_clock::now();
#endif

    if (party == ALICE) {
        // -------------------- Preparing -------------------- //
        // Receive cipher text input
        int cts_size = INPUT_DIM * COMMON_DIM / lin.data_lin1_0.slot_count;
        h1.resize(cts_size);

#ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
#endif

        recv_encrypted_vector(lin.he_8192->context, io, h1);
        cout << "> Receive input cts from client " << endl;
    } else {
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        cout << "> Repacking to column" << endl;

        // Column Packing
        vector<uint64_t> input_col(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++) {
            for (int i = 0; i < INPUT_DIM; i++) {
                input_col[j * INPUT_DIM + i] =
                    neg_mod(((int64_t)input_plain[i][j]) >> 7, (int64_t)lin.he_8192->plain_mod);
                if (prune) {
                    h1_cache_12_original[i * COMMON_DIM + j] = input_plain[i][j];
                } else {
                    h1_cache_12[i * COMMON_DIM + j] = input_plain[i][j];
                }
            }
        }

        cout << "> Send to client" << endl;

        // Send cipher text input
        vector<Ciphertext> h1_cts = lin.bert_efficient_preprocess_vec(lin.he_8192, input_col, lin.data_lin1_0);

#ifdef BERT_PERF
        t_linear1 = high_resolution_clock::now();
#endif

        send_encrypted_vector(io, h1_cts);
    }

    BFVParm *parm = new BFVParm();
    BFVKey *_party = new BFVKey();
    FPMath **fpmath = nl.fpmath;
    _party->parm = parm;
    _party->party = party;
    _party->encryptor = lin.he_8192->encryptor;
    _party->decryptor = lin.he_8192->decryptor;
    _party->parm->poly_modulus_degree = lin.he_8192->poly_modulus_degree;
    _party->parm->plain_mod = lin.he_8192->plain_mod;
    _party->parm->context = lin.he_8192->context;
    _party->parm->evaluator = lin.he_8192->evaluator;
    _party->parm->encoder = lin.he_8192->encoder;
    BFVLongCiphertext *inp_e = nullptr;
    vector<uint64_t> input(batch_size * d_module);
    memcpy(input.data(), h1_cache_12, batch_size * d_module * sizeof(uint64_t));
    if (party == BOB) {
        inp_e = RFCP_bfv_encodeA(input, _party, batch_size, d_module, d_k);
        send_encoded_ciper(inp_e, fpmath, d_module);
    } else {
        inp_e = new BFVLongCiphertext[d_module];
        recv_encoded_ciper(inp_e, fpmath, d_module, _party->parm->context);
    }
    vector<vector<uint64_t>> attn_output_h(n_heads);
    vector<uint64_t> tmp_output(batch_size * d_module, 100000);

    cout << "> --- Entering Attention Layers ---" << endl;
    for (int layer_id = 0; layer_id < ATTENTION_LAYERS; ++layer_id) {
        {
            // for (int head = 0; head < n_heads; head++) {
                vector<uint64_t> Q, K, V, wq(d_module * d_k), wk(d_module * d_k), wv(d_module * d_k),
                    bq(batch_size * d_k), bk(batch_size * d_k), bv(batch_size * d_k);
                for (int j = 0; j < d_k; j++) {
                    for (int i = 0; i < d_module; i++) {
                        // wq[i * d_k + j] = bm.w_q[layer_id][head][i][j];
                        // wk[i * d_k + j] = bm.w_k[layer_id][head][i][j];
                        // wv[i * d_k + j] = bm.w_v[layer_id][head][i][j];
                        wq[i * d_k + j] = 1000;
                        wk[i * d_k + j] = 1000;
                        wv[i * d_k + j] = 1000;
                    }
                    for (int i = 0; i < batch_size; i++) {
                        bq[i * d_k + j] = 1000;
                        bk[i * d_k + j] = 1000;
                        bv[i * d_k + j] = 1000;
                    }
                }
                // BFVLongCiphertext *Q_encode, *Q_encode_remote = new BFVLongCiphertext[d_k], *softmax_encode,
                //                              *softmax_encode_remote = new BFVLongCiphertext[batch_size];
                // if (party == ALICE) {
                // Q = conv->he_to_ss_client(fpmath[head]->iopack->io, _party);
                // K = conv->he_to_ss_client(fpmath[head]->iopack->io, _party);
                // V = conv->he_to_ss_client(fpmath[head]->iopack->io, _party);
                // } else {

                // BFVLongCiphertext Q_sec_a = RFCP_bfv_matmul(inp_e, wq, batch_size, d_module, d_k, _party->parm),
                //                   K_sec_a = RFCP_bfv_matmul(inp_e, wk, batch_size, d_module, d_k, _party->parm),
                //                   V_sec_a = RFCP_bfv_matmul(inp_e, wv, batch_size, d_module, d_k, _party->parm);
                // BFVLongPlaintext bQ_plain(_party->parm, wk), bK_plain(_party->parm, wk), bV_plain(_party->parm, wk);
                // Q_sec_a.add_plain_inplace(bQ_plain, _party->parm->evaluator);
                // K_sec_a.add_plain_inplace(bK_plain, _party->parm->evaluator);
                // V_sec_a.add_plain_inplace(bV_plain, _party->parm->evaluator);

                // Q = conv->he_to_ss_server(fpmath[head]->iopack->io, _party->parm, Q_sec_a);
                // K = conv->he_to_ss_server(fpmath[head]->iopack->io, _party->parm, K_sec_a);
                // V = conv->he_to_ss_server(fpmath[head]->iopack->io, _party->parm, K_sec_a);
                // }
                // conv->Prime_to_Ring(_party->party, V.data(), V.data(), batch_size * d_k, DEFAULT_ELL,
                //                     _party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE, fpmath[head]);
                // vector<uint64_t> K_T(batch_size * d_k);
                // for (int i = 0; i < batch_size; i++) {
                //     for (int j = 0; j < d_k; j++) {
                //         K_T[j * batch_size + i] = K[i * d_k + j];
                //     }
                // }
                // vector<uint64_t> QK_local = matmul(Q, K, batch_size, d_k, batch_size, true);
                // Q_encode = RFCP_bfv_encodeA(Q, party, batch_size, d_k, batch_size);
                // times[head] += (get_timestamp() - start_QKV);
                //         if (party->party == ALICE) {
                //             times[head] += send_encoded_ciper(Q_encode, fpmath, d_k);
                //             recv_encoded_ciper(Q_encode_remote, fpmath, d_k);
                //         } else {
                //             recv_encoded_ciper(Q_encode_remote, fpmath, d_k);
                //             times[head] += send_encoded_ciper(Q_encode, fpmath, d_k);
                //         }

                //         timestamp start_QK = get_timestamp();
                //         BFVLongCiphertext QK_enc =
                //             RFCP_bfv_matmul(Q_encode_remote, K_T, batch_size, d_k, batch_size, party->parm);
                //         vector<uint64_t> ret1, ret2;
                //         if (party->party == ALICE) {
                //             ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, QK_enc);
                //             ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
                //         } else {
                //             ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
                //             ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm, QK_enc);
                //         }
                //         for (int i = 0; i < batch_size * batch_size; i++) {
                //             QK_local[i] = (QK_local[i] + ret1[i] + ret2[i]) & (1ULL << DEFAULT_ELL);
                //         }
                //         conv->Prime_to_Ring(party->party, QK_local.data(), QK_local.data(), batch_size * batch_size,
                //                             DEFAULT_ELL, party->parm->plain_mod, DEFAULT_SCALE * 2, DEFAULT_SCALE,
                //                             fpmath[head]);
                //         vector<uint64_t> softmax_output;
                //         times[head] += get_timestamp() - start_QK;
                //         times[head] += softmax(QK_local, softmax_output, fpmath[head], conv);

                //         timestamp start_SV_local = get_timestamp();
                //         attn_output_h[head] = matmul(softmax_output, V, batch_size, batch_size, d_k);
                //         softmax_encode = RFCP_bfv_encodeA(softmax_output, party, batch_size, batch_size, d_k);
                //         times[head] += get_timestamp() - start_SV_local;
                //         if (party->party == ALICE) {
                //             recv_encoded_ciper(softmax_encode_remote, fpmath, batch_size);
                //             times[head] += send_encoded_ciper(softmax_encode, fpmath, batch_size);
                //         } else {
                //             times[head] += send_encoded_ciper(softmax_encode, fpmath, batch_size);
                //             recv_encoded_ciper(softmax_encode_remote, fpmath, batch_size);
                //         }

                //         timestamp start_SV = get_timestamp();
                //         BFVLongCiphertext attn_out_enc =
                //             RFCP_bfv_matmul(softmax_encode_remote, V, batch_size, batch_size, d_k, party->parm);
                //         vector<uint64_t> attn_out_ret1, attn_out_ret2;
                //         if (party->party == ALICE) {
                //             attn_out_ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm,
                //             attn_out_enc); attn_out_ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
                //         } else {
                //             attn_out_ret2 = conv->he_to_ss_client(fpmath[head]->iopack->io, party);
                //             attn_out_ret1 = conv->he_to_ss_server(fpmath[head]->iopack->io, party->parm,
                //             attn_out_enc);
                //         }
                //         for (int i = 0; i < batch_size * d_k; i++) {
                //             attn_output_h[head][i] =
                //                 (attn_output_h[head][i] + attn_out_ret1[i] + attn_out_ret2[i]) & (1ULL <<
                //                 DEFAULT_ELL);
                //         }
                //         times[head] += get_timestamp() - start_SV;

                //         delete[] softmax_encode_remote;
                //         delete[] softmax_encode;
                //         delete[] Q_encode_remote;
                //         delete[] Q_encode;
            // }

            // #pragma omp parallel for
            //             for (int h = 0; h < n_heads; h++) {
            //                 for (int i = 0; i < batch_size; i++) {
            //                     for (int j = 0; j < d_k; j++) {
            //                         tmp_output[i * d_module + h * d_k + j] = attn_output_h[h][i * d_k + j];
            //                     }
            //                 }
            //             }
            if (party == ALICE) {
                h2 = ss_to_he_server(lin.he_8192_tiny, tmp_output.data(), tmp_output.size(), DEFAULT_SCALE);
            } else {
                ss_to_he_client(lin.he_8192_tiny, tmp_output.data(), tmp_output.size(), DEFAULT_SCALE);
            }
        }

        // -------------------- Linear #2 -------------------- //
        {
            FCMetadata data = lin.data_lin2;

            int ln_size = data.image_size * COMMON_DIM;
            int ln_cts_size = ln_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t *ln_input_cross = new uint64_t[ln_size];
            uint64_t *ln_input_row = new uint64_t[ln_size];
            uint64_t *ln_output_row = new uint64_t[ln_size];
            uint64_t *ln_output_col = new uint64_t[ln_size];
            uint64_t *ln_wx = new uint64_t[ln_size];

            uint64_t *ln_weight = new uint64_t[ln_size];
            uint64_t *ln_bias = new uint64_t[ln_size];

            if (party == ALICE) {
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;
                vector<Ciphertext> h3 = lin.linear_2(lin.he_8192_tiny, h2, lin.pp_2[layer_id], data);
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h3, ln_input_cross, true);
                ln_share_server(layer_id, lin.w_ln_1[layer_id], lin.b_ln_1[layer_id], ln_weight, ln_bias, data);
            } else {
                vector<Ciphertext> h3(ln_cts_size);
                he_to_ss_client(lin.he_8192_tiny, ln_input_cross, ln_cts_size, lin.data_lin2);
                ln_share_client(ln_weight, ln_bias, data);
            }

#ifdef BERT_PERF
            t_total_linear2 += interval(t_linear2);

            c_linear_2 += get_comm();
            r_linear_2 += get_round();
            auto t_repacking = high_resolution_clock::now();
#endif

            lin.plain_col_packing_postprocess(ln_input_cross, ln_input_row, false, data);

#ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
#endif

            nl.gt_p_sub(NL_NTHREADS, ln_input_row, lin.he_8192_tiny->plain_mod, ln_input_row, ln_size, NL_ELL, NL_SCALE,
                        NL_SCALE);

#ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_ln_1 = high_resolution_clock::now();
#endif

            for (int i = 0; i < ln_size; i++) {
                ln_input_row[i] += h1_cache_12[i];
            }

            // Layer Norm
            nl.layer_norm(NL_NTHREADS, ln_input_row, ln_output_row, ln_weight, ln_bias, data.image_size, COMMON_DIM,
                          NL_ELL, NL_SCALE);

            // wx
            if (party == ALICE) {
                vector<Ciphertext> ln = ss_to_he_server(lin.he_8192_ln, ln_output_row, ln_size, NL_ELL);

                vector<Ciphertext> ln_w = lin.w_ln(lin.he_8192_ln, ln, lin.w_ln_1_pt[layer_id]);
                he_to_ss_server(lin.he_8192_ln, ln_w, ln_wx, true);
            } else {
                ss_to_he_client(lin.he_8192_ln, ln_output_row, ln_size, NL_ELL);
                int cts_size = ln_size / lin.he_8192_ln->poly_modulus_degree;
                he_to_ss_client(lin.he_8192_ln, ln_wx, cts_size, data);
            }

            nl.gt_p_sub(NL_NTHREADS, ln_wx, lin.he_8192_ln->plain_mod, ln_wx, ln_size, NL_ELL, 2 * NL_SCALE, NL_SCALE);

            uint64_t ell_mask = (1ULL << (NL_ELL)) - 1;

            for (int i = 0; i < ln_size; i++) {
                ln_wx[i] += ln_bias[i] & ell_mask;
            }

#ifdef BERT_PERF
            t_total_ln_1 += interval(t_ln_1);
            c_ln1 += get_comm();
            r_ln1 += get_round();
            auto t_shift = high_resolution_clock::now();
#endif

            memcpy(h4_cache_12, ln_wx, ln_size * sizeof(uint64_t));

            nl.right_shift(NL_NTHREADS, ln_wx, NL_SCALE - 5, ln_output_row, ln_size, NL_ELL, NL_SCALE);

#ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_repacking_2 = high_resolution_clock::now();
#endif

            lin.plain_col_packing_preprocess(ln_output_row, ln_output_col, lin.he_8192_tiny->plain_mod, data.image_size,
                                             COMMON_DIM);

#ifdef BERT_PERF
            t_total_repacking += interval(t_repacking_2);

            t_linear3 = high_resolution_clock::now();
#endif

            if (party == ALICE) {
                h4 = ss_to_he_server(lin.he_8192_tiny, ln_output_col, ln_size, NL_ELL);
            } else {
                ss_to_he_client(lin.he_8192_tiny, ln_output_col, ln_size, NL_ELL);
            }

            delete[] ln_input_cross;
            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_output_col;
            delete[] ln_weight;
            delete[] ln_bias;
        }

        // -------------------- Linear #3 -------------------- //
        {
            FCMetadata data = lin.data_lin3;

            int gelu_input_size = data.image_size * 3072;
            int gelu_cts_size = gelu_input_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t *gelu_input_cross = new uint64_t[gelu_input_size];
            uint64_t *gelu_input_col = new uint64_t[gelu_input_size];
            uint64_t *gelu_output_col = new uint64_t[gelu_input_size];

            if (party == ALICE) {
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;
                vector<Ciphertext> h5 = lin.linear_2(lin.he_8192_tiny, h4, lin.pp_3[layer_id], data);

                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h5, gelu_input_cross, true);
            } else {
                he_to_ss_client(lin.he_8192_tiny, gelu_input_cross, gelu_cts_size, data);
            }

#ifdef BERT_PERF
            t_total_linear3 += interval(t_linear3);

            c_linear_3 += get_comm();
            r_linear_3 += get_round();
            auto t_repacking = high_resolution_clock::now();
#endif

            lin.plain_col_packing_postprocess(gelu_input_cross, gelu_input_col, true, data);

#ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
#endif

            // mod p
            nl.gt_p_sub(NL_NTHREADS, gelu_input_col, lin.he_8192_tiny->plain_mod, gelu_input_col, gelu_input_size,
                        GELU_ELL, 11, GELU_SCALE);

#ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_gelu = high_resolution_clock::now();
#endif

            nl.gelu(NL_NTHREADS, gelu_input_col, gelu_output_col, gelu_input_size, GELU_ELL, GELU_SCALE);

#ifdef BERT_PERF
            t_total_gelu += interval(t_gelu);
            c_gelu += get_comm();
            r_gelu += get_round();
            auto t_shift = high_resolution_clock::now();
#endif

#ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();

            t_linear4 = high_resolution_clock::now();
#endif

            if (party == ALICE) {
                h6 = ss_to_he_server(lin.he_8192_tiny, gelu_output_col, gelu_input_size, NL_ELL);
            } else {
                ss_to_he_client(lin.he_8192_tiny, gelu_output_col, gelu_input_size, NL_ELL);
            }

            delete[] gelu_input_cross;
            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        {
            FCMetadata data = lin.data_lin4;

            int ln_2_input_size = data.image_size * COMMON_DIM;
            int ln_2_cts_size = ln_2_input_size / lin.he_8192_tiny->poly_modulus_degree;

            uint64_t *ln_2_input_cross = new uint64_t[ln_2_input_size];
            uint64_t *ln_2_input_row = new uint64_t[ln_2_input_size];
            uint64_t *ln_2_output_row = new uint64_t[ln_2_input_size];
            uint64_t *ln_2_output_col = new uint64_t[ln_2_input_size];

            uint64_t *ln_2_wx = new uint64_t[ln_2_input_size];

            uint64_t *ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t *ln_bias_2 = new uint64_t[ln_2_input_size];

            if (party == ALICE) {
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;

                vector<Ciphertext> h7 = lin.linear_2(lin.he_8192_tiny, h6, lin.pp_4[layer_id], data);

                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;
                he_to_ss_server(lin.he_8192_tiny, h7, ln_2_input_cross, true);
                ln_share_server(layer_id, lin.w_ln_2[layer_id], lin.b_ln_2[layer_id], ln_weight_2, ln_bias_2, data);
            } else {
                he_to_ss_client(lin.he_8192_tiny, ln_2_input_cross, ln_2_cts_size, data);
                ln_share_client(ln_weight_2, ln_bias_2, data);
            }

#ifdef BERT_PERF
            t_total_linear4 += interval(t_linear4);

            c_linear_4 += get_comm();
            r_linear_4 += get_round();
            auto t_repacking = high_resolution_clock::now();
#endif
            // Post Processing
            lin.plain_col_packing_postprocess(ln_2_input_cross, ln_2_input_row, false, data);

#ifdef BERT_PERF
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
#endif

            // mod p
            if (layer_id == 9 || layer_id == 10) {
                nl.gt_p_sub(NL_NTHREADS, ln_2_input_row, lin.he_8192_tiny->plain_mod, ln_2_input_row, ln_2_input_size,
                            NL_ELL, 8, NL_SCALE);
            } else {
                nl.gt_p_sub(NL_NTHREADS, ln_2_input_row, lin.he_8192_tiny->plain_mod, ln_2_input_row, ln_2_input_size,
                            NL_ELL, 9, NL_SCALE);
            }

#ifdef BERT_PERF
            t_total_gt_sub += interval(t_gt_sub);
            c_gt_sub += get_comm();
            r_gt_sub += get_round();
            auto t_ln = high_resolution_clock::now();
#endif

            for (int i = 0; i < ln_2_input_size; i++) {
                ln_2_input_row[i] += h4_cache_12[i];
            }

            nl.layer_norm(NL_NTHREADS, ln_2_input_row, ln_2_output_row, ln_weight_2, ln_bias_2, data.image_size,
                          COMMON_DIM, NL_ELL, NL_SCALE);

            // wx
            if (party == ALICE) {
                vector<Ciphertext> ln = ss_to_he_server(lin.he_8192_ln, ln_2_output_row, ln_2_input_size, NL_ELL);
                vector<Ciphertext> ln_w = lin.w_ln(lin.he_8192_ln, ln, lin.w_ln_2_pt[layer_id]);
                he_to_ss_server(lin.he_8192_ln, ln_w, ln_2_wx, true);
            } else {
                ss_to_he_client(lin.he_8192_ln, ln_2_output_row, ln_2_input_size, NL_ELL);
                int cts_size = ln_2_input_size / lin.he_8192_ln->poly_modulus_degree;
                he_to_ss_client(lin.he_8192_ln, ln_2_wx, cts_size, data);
            }

            nl.gt_p_sub(NL_NTHREADS, ln_2_wx, lin.he_8192_ln->plain_mod, ln_2_wx, ln_2_input_size, NL_ELL, 2 * NL_SCALE,
                        NL_SCALE);

            uint64_t ell_mask = (1ULL << (NL_ELL)) - 1;

            for (int i = 0; i < ln_2_input_size; i++) {
                ln_2_wx[i] += ln_bias_2[i] & ell_mask;
            }

#ifdef BERT_PERF
            t_total_ln_2 += interval(t_ln);
            c_ln2 += get_comm();
            r_ln2 += get_round();
            auto t_shift = high_resolution_clock::now();
#endif

            // update H1
            memcpy(h1_cache_12, ln_2_wx, ln_2_input_size * sizeof(uint64_t));

            // Rescale
            nl.right_shift(NL_NTHREADS, ln_2_wx, 12 - 5, ln_2_output_row, ln_2_input_size, NL_ELL, NL_SCALE);

#ifdef BERT_PERF
            t_total_shift += interval(t_shift);
            c_shift += get_comm();
            r_shift += get_round();
            auto t_repacking_2 = high_resolution_clock::now();
#endif

            lin.plain_col_packing_preprocess(ln_2_output_row, ln_2_output_col, lin.he_8192_tiny->plain_mod,
                                             data.image_size, COMMON_DIM);

#ifdef BERT_PERF
            t_total_repacking += interval(t_repacking_2);

            t_linear1 = high_resolution_clock::now();
#endif

            if (layer_id == 11) {
                // Using Scale of 12 as
                memcpy(h98, h1_cache_12, COMMON_DIM * sizeof(uint64_t));
            } else {
                if (party == ALICE) {
                    h1 = ss_to_he_server(lin.he_8192, ln_2_output_col, ln_2_input_size, NL_ELL);
                } else {
                    ss_to_he_client(lin.he_8192, ln_2_output_col, ln_2_input_size, NL_ELL);
                }
            }

            delete[] ln_2_input_cross;
            delete[] ln_2_input_row;
            delete[] ln_2_output_row;
            delete[] ln_2_output_col;
            delete[] ln_weight_2;
            delete[] ln_bias_2;
        }
    }

#ifdef BERT_PERF
    c_pc += get_comm();
    r_pc += get_round();
    // cout << "> [TIMING]: linear1 takes " << t_total_linear1 << " sec" << endl;
    cout << "> [TIMING]: linear2 takes " << t_total_linear2 << " sec" << endl;
    cout << "> [TIMING]: linear3 takes " << t_total_linear3 << " sec" << endl;
    cout << "> [TIMING]: linear4 takes " << t_total_linear4 << " sec" << endl;

    // cout << "> [TIMING]: softmax takes " << t_total_softmax << " sec" << endl;
    // cout << "> [TIMING]: pruning takes " << t_total_pruning << " sec" << endl;
    // cout << "> [TIMING]: mul v takes " << t_total_mul_v << " sec" << endl;
    // cout << "> [TIMING]: gelu takes " << t_total_gelu << " sec" << endl;
    // cout << "> [TIMING]: ln_1 takes " << t_total_ln_1 << " sec" << endl;
    // cout << "> [TIMING]: ln_2 takes " << t_total_ln_2 << " sec" << endl;

    // cout << "> [TIMING]: repacking takes " << t_total_repacking << " sec" << endl;
    // cout << "> [TIMING]: gt_sub takes " << t_total_gt_sub << " sec" << endl;
    // cout << "> [TIMING]: shift takes " << t_total_shift << " sec" << endl;

    // cout << "> [TIMING]: conversion takes " << t_total_conversion << " sec" << endl;
    // cout << "> [TIMING]: ln_share takes " << t_total_ln_share << " sec" << endl;

    // cout << "> [NETWORK]: Linear 1 consumes: " << c_linear_1 << " bytes" << endl;
    cout << "> [NETWORK]: Linear 2 consumes: " << c_linear_2 << " bytes" << endl;
    cout << "> [NETWORK]: Linear 3 consumes: " << c_linear_3 << " bytes" << endl;
    cout << "> [NETWORK]: Linear 4 consumes: " << c_linear_4 << " bytes" << endl;

    // cout << "> [NETWORK]: Softmax consumes: " << c_softmax << " bytes" << endl;
    // cout << "> [NETWORK]: GELU consumes: " << c_gelu << " bytes" << endl;
    // cout << "> [NETWORK]: Layer Norm 1 consumes: " << c_ln1 << " bytes" << endl;
    // cout << "> [NETWORK]: Layer Norm 2 consumes: " << c_ln2 << " bytes" << endl;
    // // cout << "> [NETWORK]: Tanh consumes: " << c_tanh << " bytes" << endl;

    // cout << "> [NETWORK]: Softmax * V: " << c_softmax_v << " bytes" << endl;
    // cout << "> [NETWORK]: Pruning: " << c_pruning << " bytes" << endl;
    // cout << "> [NETWORK]: Shift consumes: " << c_shift << " bytes" << endl;
    // cout << "> [NETWORK]: gt_sub consumes: " << c_gt_sub << " bytes" << endl;

    // cout << "> [NETWORK]: Pooling / C consumes: " << c_pc << " bytes" << endl;

    // cout << "> [NETWORK]: Linear 1 consumes: " << r_linear_1 << " rounds" << endl;
    cout << "> [NETWORK]: Linear 2 consumes: " << r_linear_2 << " rounds" << endl;
    cout << "> [NETWORK]: Linear 3 consumes: " << r_linear_3 << " rounds" << endl;
    cout << "> [NETWORK]: Linear 4 consumes: " << r_linear_4 << " rounds" << endl;

    // cout << "> [NETWORK]: Softmax consumes: " << r_softmax << " rounds" << endl;
    // cout << "> [NETWORK]: GELU consumes: " << r_gelu << " rounds" << endl;
    // cout << "> [NETWORK]: Layer Norm 1 consumes: " << r_ln1 << " rounds" << endl;
    // cout << "> [NETWORK]: Layer Norm 2 consumes: " << r_ln2 << " rounds" << endl;

    // cout << "> [NETWORK]: Softmax * V: " << r_softmax_v << " rounds" << endl;
    // cout << "> [NETWORK]: Pruning: " << r_pruning << " rounds" << endl;
    // cout << "> [NETWORK]: Shift consumes: " << r_shift << " rounds" << endl;
    // cout << "> [NETWORK]: gt_sub consumes: " << r_gt_sub << " rounds" << endl;
#endif

    if (party == ALICE) {
        return {};
    } else {
        return vector<double>(NUM_CLASS, .5);
    }
}