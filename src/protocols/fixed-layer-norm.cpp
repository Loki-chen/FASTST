#include "fixed-layer-norm.h"

BFVLongCiphertext FixedLayerNorm::forward(const BFVLongCiphertext &attn, const bfv_matrix &input) const
{

    PRG128 prg;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(-1, 1);
    size_t i, j;

    if (party->party == ALICE)
    {

        // generate ha1, ha2, commpute ha1xa, ha2/ha1, [attn]_b * ha2, encrypt [ha2/ha1]_a, [ha2]_a
        const double ha1 = dist(gen), ha2 = dist(gen), ka = dist(gen);
        const double ha2_div_ha1 = ha2 / ha1, div_ha2 = 1 / ha2;

        const uint64_t uint_ha1 = sci::neg_mod(static_cast<int64_t>(ha1 * (1ULL << DEFAULT_SCALE)), DEFAULT_ELL);
        const uint64_t uint_ha2 = sci::neg_mod(static_cast<int64_t>(ha2 * (1ULL << DEFAULT_SCALE)), DEFAULT_ELL);
        const uint64_t uint_ha2_div_ha1 = sci::neg_mod(static_cast<int64_t>(ha2_div_ha1 * (1ULL << DEFAULT_SCALE)), DEFAULT_ELL);
        const uint64_t uint_div_ha2 = sci::neg_mod(static_cast<int64_t>(div_ha2 * (1ULL << DEFAULT_SCALE)), DEFAULT_ELL);

        FixArray fix_ha1 = fpmath->fix->input(sci::ALICE, input.size(), uint_ha1, true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_ha2 = fpmath->fix->input(sci::ALICE, input.size(), uint_ha2, true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_ha2_div_ha1 = fpmath->fix->input(sci::ALICE, input.size(), uint_ha2_div_ha1, true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_div_ha2 = fpmath->fix->input(sci::ALICE, input.size(), uint_div_ha2, true, DEFAULT_ELL, DEFAULT_SCALE);

        FixArray fix_input = fpmath->fix->input(party->party, input.size(), input.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        BFVLongPlaintext ha2_plain(parm, fix_ha2.data, fix_ha2.size);
        // Need fix BFVLongPlaintext encrypt for *data
        BFVLongCiphertext ha2_div_ha1_secret_a(parm, fix_ha2_div_ha1.data, fix_ha2_div_ha1.size, party), ha2_secret_a(ha2_plain, party), xha1_secret_a;
        FixArray fixed_ha1_xa = fpmath->fix->public_mul(fix_input, fix_ha1, DEFAULT_ELL + 2 * DEFAULT_SCALE);
        BFVLongCiphertext attn_ha2_b = attn.multiply_plain(ha2_plain, party->parm->evaluator);

        // send H5 = {ha1_xa, ha2_div_hc1_secret_a, ha2_secret_a, attn_ha2_b} to bob
        fpmath_public->fix->send_fix_array(fixed_ha1_xa);
        BFVLongCiphertext::send(fpmath_public->fix->iopack->io, &ha2_div_ha1_secret_a);
        BFVLongCiphertext::send(fpmath_public->fix->iopack->io, &ha2_secret_a);
        BFVLongCiphertext::send(fpmath_public->fix->iopack->io, &attn_ha2_b); // error

        /*
            alice receive H2, and get x * gb
            1. compute mean(x * gb) = gb * \mu, standard_deviation(x * gb, gb * \mu) = sigma * gs
            2. generate kc, compute tmp1 = (x * gb - gb * \mu) * kc = (x - \mu)*gb*kc, tmp2_secret_c = [1 / (sigma * gs * ka)]_c
        */
        BFVLongCiphertext::recv(fpmath_public->fix->iopack->io, &xha1_secret_a, parm->context);
        // cout << "test \n";
        BFVLongPlaintext ha2xgb_plain = xha1_secret_a.decrypt(party);

        bfv_matrix ha2xgb = ha2xgb_plain.decode(parm);
        FixArray fix_ha2xgb = fpmath->fix->input(sci::PUBLIC, ha2xgb.size(), ha2xgb.data(), true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_xgb = fpmath->fix->mul(fix_ha2xgb, fix_div_ha2);
        fix_xgb.party = PUBLIC;
        fix_xgb = fpmath->fix->location_truncation(fix_xgb, DEFAULT_SCALE);
        fix_xgb.party = sci::ALICE;
        // auto fix = fpmath->fix;
        // print_fix(fix_xgb);
        // FixArray temp(fix_xgb.party, fix_xgb.size / batch_size, fix_xgb.signed_, fix_xgb.ell, fix_xgb.s);
        vector<FixArray> fix_xgb_array(batch_size);
        for (i = 0; i < batch_size; i++)
        {
            fix_xgb_array[i] = FixArray(fix_xgb.party, d_module, fix_xgb.signed_, fix_xgb.ell, fix_xgb.s);
            for (j = 0; j < d_module; j++)
            {
                fix_xgb_array[i].data[j] = fix_xgb.data[i * d_module + j];
            }
            fix_xgb_array.push_back(fix_xgb_array[i]);
        }
        vector<FixArray> out_array = fpmath->mean(fix_xgb_array);

        // vector<FixArray> mu_gb(fix_xgb.party, batch_size, fix_xgb.signed_, fix_xgb.ell, fix_xgb.s);
        // for (i = 0; i < batch_size; i++)
        // {
        //     for (j = 0; j < d_module; j++)
        //     {
        //         mu_gb.data[j] = fix_xgb.data[j];
        //     }
        // }
    }
    else
    {
        const double gb = dist(gen);
        const uint64_t fix_gb = sci::neg_mod(static_cast<int64_t>(gb * (1ULL << DEFAULT_SCALE)), DEFAULT_ELL);
        bfv_matrix gamma(batch_size * d_module), beta(batch_size * d_module);
        FixArray ha1_xa, tmp1;
        BFVLongPlaintext input_b_plain(parm, input), gb_plain(parm, fix_gb);
        BFVLongCiphertext ha2_div_ha1_secret_a, ha2_secret_a, attn_ha2_b, tmp2_secret_a;
        random_bfv_mat(gamma);
        random_bfv_mat(beta);

        fpmath_public->fix->recv_fix_array(ha1_xa);
        BFVLongCiphertext::recv(fpmath_public->fix->iopack->io, &ha2_div_ha1_secret_a, parm->context);
        BFVLongCiphertext::recv(fpmath_public->fix->iopack->io, &ha2_secret_a, parm->context);
        BFVLongCiphertext::recv(fpmath_public->fix->iopack->io, &attn_ha2_b, parm->context);
        BFVLongPlaintext attn_ha2_plain = attn_ha2_b.decrypt(party);
        BFVLongCiphertext xha1_secret_a = ha2_secret_a.multiply_plain(input_b_plain, parm->evaluator);
        // attn_ha2_plain.mod_switch_to_inplace(xha1_secret_a.parms_id(), parm->evaluator);
        xha1_secret_a.add_plain_inplace(attn_ha2_plain, parm->evaluator);

        BFVLongPlaintext ha1_xc_plain(parm, ha1_xa.data, ha1_xa.size);
        ha2_div_ha1_secret_a.multiply_plain_inplace(ha1_xc_plain, parm->evaluator);
        // ha2_div_ha1_secret_a.mod_switch_to_inplace(xha1_secret_a.parms_id(), parm->evaluator);
        xha1_secret_a.add_inplace(ha2_div_ha1_secret_a, parm->evaluator);
        // gb_plain.mod_switch_to_inplace(xha1_secret_a.parms_id(), parm->evaluator);
        xha1_secret_a.multiply_plain_inplace(gb_plain, parm->evaluator);
        // send H2 = {xha1_secret_a} to alice;
        BFVLongCiphertext::send(fpmath_public->fix->iopack->io, &xha1_secret_a);
    }
}