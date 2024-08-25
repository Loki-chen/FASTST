#include "conversion.h"
#include "Utils/net_io_channel.h"
#include "utils/he-bfv.h"

bfv_matrix Conversion::he_to_ss_client(sci::NetIO *io, BFVKey *party) {
    BFVLongCiphertext lct;
    BFVLongCiphertext::recv(io, &lct, party->parm->context);
    auto lpt = lct.decrypt(party);
    return lpt.decode_uint(party->parm);
}

bfv_matrix Conversion::he_to_ss_server(sci::NetIO *io, BFVParm *parm, const BFVLongCiphertext &in) {
    int len = in.len;
    int slot_count = parm->poly_modulus_degree;
    bfv_matrix output(len);
    random_modP_mat(output, parm->plain_mod);
    BFVLongPlaintext output_plain(parm, output);
    BFVLongCiphertext cli_data = in.sub_plain(output_plain, parm->evaluator);
    BFVLongCiphertext::send(io, &cli_data);
    return output;
}

BFVLongCiphertext Conversion::ss_to_he_server(BFVParm *parm, NetIO *io, uint64_t *input, int length, int ell,
                                              bool is_add_share) {
    vector<uint64_t> tmp(length);
    for (int i = 0; i < length; i++) {
        tmp[i] = neg_mod(signed_val(input[i], ell), (int64_t)parm->plain_mod);
    }
    BFVLongPlaintext share_server(parm, tmp);
    BFVLongCiphertext share_client;
    BFVLongCiphertext::recv(io, &share_client, parm->context);
    is_add_share ? share_client.add_plain_inplace(share_server, parm->evaluator)
                 : share_client.multiply_plain_inplace(share_server, parm->evaluator);
    return share_client;
}

void Conversion::ss_to_he_client(BFVKey *party, NetIO *io, uint64_t *input, int length, int ell) {
    vector<uint64_t> tmp(length);
    for (int i = 0; i < length; i++) {
        tmp[i] = neg_mod(signed_val(input[i], ell), (int64_t)party->parm->plain_mod);
    }
    BFVLongPlaintext share_client_plain(party->parm, tmp);
    BFVLongCiphertext share_client(share_client_plain, party);
    BFVLongCiphertext::send(io, &share_client);
}

void Conversion::Ring_to_Prime(uint64_t input, uint64_t output, int ell, int64_t plain_mod) {
#ifdef LOG
    auto t_conversion = high_resolution_clock::now();
#endif
    output = sci::neg_mod(sci::signed_val(input, ell), (int64_t)plain_mod);

#ifdef LOG
    t_total_conversion += interval(t_conversion);
#endif
}

// R-to-P:location conversion
void Conversion::Ring_to_Prime(const uint64_t *input, uint64_t *output, int length, int ell, int64_t plain_mod) {
#ifdef LOG
    auto t_conversion = high_resolution_clock::now();
#endif

    vector<uint64_t> tmp(length);
    for (size_t i = 0; i < length; i++) {
        tmp[i] = sci::neg_mod(sci::signed_val(input[i], ell), (int64_t)plain_mod);
    }
    memcpy(output, tmp.data(), length * sizeof(uint64_t));

#ifdef LOG
    t_total_conversion += interval(t_conversion);
#endif
}
// P-to-R
void Conversion::Prime_to_Ring(int party, const uint64_t *input, uint64_t *output, int length, int ell,
                               int64_t plain_prime, int s_in, int s_out, FPMath *fpmath) {
    // if input > plain_prime, then sub plain_prime
    // sub plain_prime/2 anyway

    FixArray fix_input = fpmath->fix->input(party, length, input, true, ell, s_in);
    FixArray p_array = fpmath->fix->input(sci::PUBLIC, length, plain_prime, true, ell, s_in);
    FixArray p_2_array = fpmath->fix->input(sci::PUBLIC, length, (plain_prime - 1) / 2, true, ell, s_in);
    FixArray tmp = fpmath->gt_p_sub(fix_input, p_array);
    // tmp = fpmath->fix->sub(tmp, p_2_array);

    if (s_in > s_out) {
        tmp = fpmath->fix->right_shift(tmp, s_in - s_out);
    } else if (s_in < s_out) {
        tmp = fpmath->fix->mul(tmp, 1 << (s_out - s_in));
    }

    memcpy(output, tmp.data, length * sizeof(uint64_t));
}

void gt_p_sub_thread(int party, const uint64_t *x, uint64_t p, uint64_t *y, int num_ops, int ell, int s_in, int s_out,
                     FPMath *fpmath) {
    int this_party = party;
    FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s_in);
    FixArray p_array = fpmath->fix->input(PUBLIC, num_ops, p, true, ell, s_in);
    FixArray p_2_array = fpmath->fix->input(PUBLIC, num_ops, (p - 1) / 2, true, ell, s_in);
    FixArray output = fpmath->gt_p_sub(input, p_array);
    output = fpmath->fix->sub(output, p_2_array);

    if (s_in > s_out) {
        output = fpmath->fix->right_shift(output, s_in - s_out);
    } else if (s_in < s_out) {
        output = fpmath->fix->mul(output, 1 << (s_out - s_in));
    }

    memcpy(y, output.data, num_ops * sizeof(uint64_t));
}

void Conversion::Prime_to_Ring(int party, int nthreads, const uint64_t *input, uint64_t *output, int length, int ell,
                               int64_t plain_prime, int s_in, int s_out, FPMath **fpmath) {
    std::thread threads[nthreads];
    int chunk_size = length / nthreads;
    for (int i = 0; i < nthreads; i++) {
        int offset = i * chunk_size;
        int lnum_ops = (i == (nthreads - 1)) ? (length - offset) : chunk_size;
        threads[i] = std::thread(gt_p_sub_thread, party, input + offset, plain_prime, output + offset, lnum_ops, ell,
                                 s_in, s_out, fpmath[i]);
    }
    for (int i = 0; i < nthreads; i++) {
        threads[i].join();
    }
}