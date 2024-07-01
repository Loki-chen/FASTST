#include "fixed-layer-norm.h"
#include "FixedPoint/fixed-point.h"
#include "utils/he-bfv.h"

BFVLongCiphertext FixedLayerNorm::forward(const BFVLongCiphertext &attn,
                                          const bfv_matrix &input) const {

    sci::PRG128 prg;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-1, 1);
    size_t i, j;

    uint64_t *x = new uint64_t[input.size()];
    for (size_t i = 0; i < input.size(); i++)
    {
        x[i] = input[i];
    }

    if (party->party == ALICE)
    {
        /*
        Alice generate ha
        compute xa_ha,  [ha], attn_sec_b_ ha
        */

        double ha = dist(gen);
        uint64_t *prime_ha_xa = new uint64_t[batch_size * d_module];
        uint64_t *prime_ha = new uint64_t[batch_size * d_module];
        FixArray fix_ha = fpmath->fix->input(sci::ALICE, batch_size * d_module,
                                             (sci::neg_mod(static_cast<int64_t>(ha * (1ULL << (DEFAULT_SCALE))), DEFAULT_ELL)),
                                             true, DEFAULT_ELL, DEFAULT_SCALE);

        FixArray fix_div_ha = fpmath->fix->input(sci::ALICE, batch_size * d_module,
                                                 (sci::neg_mod(static_cast<int64_t>((1.0 / ha) * (1ULL << (DEFAULT_SCALE))), DEFAULT_ELL)),
                                                 true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_xa = fpmath->fix->input(sci::ALICE, batch_size * d_module,
                                             x,
                                             true, DEFAULT_ELL, DEFAULT_SCALE);

#ifdef LOG
        INIT_TIMER
        START_TIMER
#endif
        fix_ha.party = sci::PUBLIC; // just to make the mul useful.
        FixArray fix_ha_xa = fpmath->fix->mul(fix_xa, fix_ha, DEFAULT_ELL);
        fix_ha_xa = fpmath->fix->location_truncation(fix_ha_xa, DEFAULT_SCALE);
        conv->Ring_to_Prime(fix_ha_xa.data, prime_ha_xa, batch_size * d_module, DEFAULT_ELL, parm->plain_mod);
        conv->Ring_to_Prime(fix_ha.data, prime_ha, batch_size * d_module, DEFAULT_ELL, parm->plain_mod);
        BFVLongPlaintext ha_plain(parm, prime_ha_xa, batch_size * d_module);
        BFVLongCiphertext ha_secret_a(ha_plain, party);
        BFVLongCiphertext attn_ha_secret_b = attn.multiply_plain(ha_plain, parm->evaluator);
        attn_ha_secret_b.mod_switch_to_next_inplace(parm->evaluator);

        // Alice : send H1 = {ha_xa, ha_secret_a, attn_ha_secret_b} to bob
        send_mat(io, fix_ha_xa.data, fix_ha_xa.size);
        BFVLongCiphertext::send(io, &ha_secret_a);
        BFVLongCiphertext::send(io, &attn_ha_secret_b);
        std::cout << "test \n";
        /*
        alice receive H2, and get x * gb
        1. compute mean(x * gb) = gb * \mu, standard_deviation(x * gb, gb * \mu) = sigma * gs
        2. generate kc, compute tmp1 = (x * gb - gb * \mu) * kc = (x - \mu)*gb*kc, tmp2_secret_c = [1 / (sigma * gs * ka)]_c
        */
        return BFVLongCiphertext();
    }
    else
    {
        double gb = dist(gen);
        FixArray fix_ha_xa = fpmath->fix->input(sci::ALICE, batch_size * d_module, x, true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongCiphertext ha_secret_a, attn_ha_secret_b;

        recv_mat(io, fix_ha_xa.data, fix_ha_xa.size);
        BFVLongCiphertext::recv(io, &ha_secret_a, parm->context);
        BFVLongCiphertext::recv(io, &attn_ha_secret_b, parm->context);

        return ha_secret_a;
    }
    delete[] x;
}