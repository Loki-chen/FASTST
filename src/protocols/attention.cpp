#include "attention.h"
Attention::Attention(CKKSKey *party_, SEALContext *context_, IOPack *io_pack_,
                     size_t d_module_, size_t d_k_, size_t head_) : party(party_),
                                                                    context(context_),
                                                                    io_pack(io_pack_),
                                                                    d_module(d_module_),
                                                                    d_k(d_k_),
                                                                    head(head_)
{
    encoder = new CKKSEncoder(*(party->context));
    evaluator = new Evaluator(*(party->context));
}

Attention::~Attention()
{
    delete encoder;
    delete evaluator;
}

LongCiphertext Attention::forward(const std::vector<double> &input)
{
    size_t i, j;
    size_t inp_seq = input.size() / d_module;
    std::vector<double> WQ(d_module * d_k), WK(d_module * d_k), WV(d_module * d_k), Attn(inp_seq * d_k);
    LongCiphertext output;
    load_mat(WQ, "WQ");
    load_mat(WK, "WK");
    load_mat(WV, "WV");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    if (party->party == ALICE)
    {
        // Alice: possess: x_b, W_b
        double ra = dist(gen);
        std::vector<double> ra_xa(inp_seq * d_module);
        for (i = 0; i < inp_seq * d_k; i++)
        {
            ra_xa[i] = ra * input[i];
        }
        for (i = 0; i < d_module * d_k; i++)
        {
            WQ[i] = ra * WQ[i];
            WK[i] = ra * WK[i];
            WV[i] = ra * WV[i];
        }
        auto ra_xa_WQa = matmul(input, WQ, inp_seq, d_module, d_k);
        auto ra_xa_WKa = matmul(input, WK, inp_seq, d_module, d_k);
        auto ra_xa_WVa = matmul(input, WV, inp_seq, d_module, d_k);
        Plaintext ra_plain;
        encoder->encode(ra, scale, ra_plain);
        Ciphertext ra_secret_a_;
        party->encryptor->encrypt(ra_plain, ra_secret_a_);
        LongCiphertext ra_secret_a(ra_secret_a_);
        send_mat(io_pack, &ra_xa_WQa);
        send_mat(io_pack, &ra_xa_WKa);
        send_mat(io_pack, &ra_xa_WVa);
        send_mat(io_pack, &ra_xa);
        send_mat(io_pack, &WQ);
        send_mat(io_pack, &WK);
        send_mat(io_pack, &WV);
        LongCiphertext::send(io_pack, &ra_secret_a);
        // send H1 = {ra_xa_WIa, ra_xa, ra_WIa, [ra]_a} to bob, where I = Q, K, V

        /*
            Alice receive H2 = {raQ_sec_a, raK_sec_a, rb1_square_secret_b}, and get Q/rs1, K/rs1, [rb1]_s
            1. compute [Z]_s = [Q .* K^T/sqrt(d_k)]_s = Q/rs1 .* K/rs1 * [rs1^2]_s / sqrt(d_k)
            2. generate Zc, compute [Zs]_s = [Z]_s - Zc, exp(Zc)
        */
        LongCiphertext raQ_sec_a, raK_sec_a, rb1_square_secret_b;
        LongCiphertext::recv(io_pack, &raQ_sec_a, context);
        LongCiphertext::recv(io_pack, &raK_sec_a, context);
        LongCiphertext::recv(io_pack, &rb1_square_secret_b, context);
        LongPlaintext raQ_div_rb1_plain = raQ_sec_a.decrypt(party);
        LongPlaintext raK_div_rb1_plain = raK_sec_a.decrypt(party);
        std::vector<double> Q_div_rb1 = raQ_div_rb1_plain.decode(encoder);
        std::vector<double> K_div_rb1 = raK_div_rb1_plain.decode(encoder);
        std::vector<double> eZa(inp_seq * inp_seq);
        random_mat(eZa, -10, 0);
        std::vector<double> negZa(eZa);
        auto sqrt_d_k = sqrt(d_k);
        for (size_t i = 0; i < inp_seq * d_k; i++)
        {
            Q_div_rb1[i] /= ra;
            Q_div_rb1[i] /= sqrt_d_k;
            K_div_rb1[i] /= ra;
        }
        auto temp_z = matmul(Q_div_rb1, K_div_rb1, inp_seq, d_k, inp_seq, true);
        for (size_t i = 0; i < inp_seq * inp_seq; i++)
        {
            negZa[i] = -negZa[i];
            eZa[i] = exp(eZa[i]);
        }
        norm(temp_z, inp_seq, inp_seq);
        LongPlaintext z_plain(temp_z, scale, party->slot_count, encoder);
        auto Zb_secret_b = rb1_square_secret_b.multiply_plain(z_plain, evaluator);
        Zb_secret_b.rescale_to_next_inplace(evaluator);
        Zb_secret_b.scale(scale);
        LongPlaintext negZc_plain(negZa, scale, party->slot_count, encoder);
        negZc_plain.mod_switch_to_inplace(Zb_secret_b.parms_id(), evaluator);
        Zb_secret_b.add_plain_inplace(negZc_plain, evaluator);
#ifdef SOFTMAX_TIME_TEST
        INIT_TIMER;
        START_TIMER;
#endif
        LongPlaintext eZa_plain(eZa, scale, party->slot_count, encoder);
        LongCiphertext eZa_secret_a_(eZa_plain, party);
        LongCiphertext::send(io_pack, &Zb_secret_b);
        LongCiphertext::send(io_pack, &eZa_secret_a_);
        // send H3 = {Zb_secret_b, eZa_secret_a} to Bob

        /*
            Alice receive H4 = {eZa_secret_a, eZb, raV_sec_a, [Attn]_s}, and get rb2 * exp(Z) + O, Db * exp(Zb), Rb * V,
            1. compute sum_j (rs2_expZ + O)_ij, (Db * exp(Zb) * exp(Za)) .* Rb * V
            2.

        */
        LongCiphertext eZa_secret_a, raV_sec_a;
        std::vector<double> eZb(inp_seq * inp_seq);
        LongCiphertext::recv(io_pack, &eZa_secret_a, context);
        recv_mat(io_pack, &eZb);
        LongCiphertext::recv(io_pack, &raV_sec_a, context);
        LongCiphertext::recv(io_pack, &output, context);

        LongPlaintext rs2_expZ_plain = eZa_secret_a.decrypt(party);
        auto rs2_expZ = rs2_expZ_plain.decode(encoder);

        LongPlaintext Rb_V_plain = raV_sec_a.decrypt(party);
        auto Rb_V = Rb_V_plain.decode(encoder);
        for (size_t i = 0; i < inp_seq * d_k; i++)
            Rb_V[i] /= ra;

        std::vector<double> exp_sum(inp_seq);
        for (size_t i = 0; i < inp_seq; i++)
        {
            for (j = 0; j < inp_seq; j++)
            {
                exp_sum[i] += rs2_expZ[i * inp_seq + j];
            }
        }
        for (i = 0; i < inp_seq; i++)
        {
            for (j = 0; j < inp_seq; j++)
            {
                eZb[i * inp_seq + j] *= eZa[i * inp_seq + j];
                eZb[i * inp_seq + j] /= exp_sum[i];
            }
        }
#ifdef SOFTMAX_TIME_TEST
        STOP_TIMER("attention softmax");
#endif
        auto Attn = matmul(eZb, Rb_V, inp_seq, inp_seq, d_k);
        LongPlaintext Attn_plain(Attn, scale, party->slot_count, encoder);
        output.multiply_plain_inplace(Attn_plain, evaluator);
        output.rescale_to_next_inplace(evaluator);
        output.scale(scale);
    }
    else if (party->party == BOB)
    {
        /*
            Bob: revice H1 = {ra_xa_WIa, ra_xa, ra_WIa, [ra]_a}, and possess: x_b, W_b
            1. compute: rxw_a + rx_a * w_b + rW_a * x_b + [r_a]_a * xw_b = [r_aI]_a , where I stands for  Q,K,V
            2. genereat random num r_b, compute [r_aQ/r_b]_a, [r_aK/r_b]_a, [(r_b)^2]_b
        */
        std::vector<double> ra_xa_WQa(inp_seq * d_k),
            ra_xa_WKa(inp_seq * d_k),
            ra_xa_WVa(inp_seq * d_k),
            ra_xa(inp_seq * d_module),
            ra_WQa(d_module * d_k),
            ra_WKa(d_module * d_k),
            ra_WVa(d_module * d_k);
        LongCiphertext ra_secret_a;
        recv_mat(io_pack, &ra_xa_WQa);
        recv_mat(io_pack, &ra_xa_WKa);
        recv_mat(io_pack, &ra_xa_WVa);
        recv_mat(io_pack, &ra_xa);
        recv_mat(io_pack, &ra_WQa);
        recv_mat(io_pack, &ra_WKa);
        recv_mat(io_pack, &ra_WVa);
        LongCiphertext::recv(io_pack, &ra_secret_a, context);
        auto cal_raI_A = [](std::vector<double> input_b, std::vector<double> WIb,
                            std::vector<double> ra_xa, std::vector<double> ra_WIa, std::vector<double> ra_xa_WIa,
                            LongCiphertext ra_secret_a,
                            CKKSKey *party, CKKSEncoder *encoder, Evaluator *evaluator,
                            double scale, size_t inp_seq, size_t d_module, size_t d_k)
        {
            auto xbWI_b = matmul(input_b, WIb, inp_seq, d_module, d_k);
            LongPlaintext xbWI_b_plain(xbWI_b, scale, party->slot_count, encoder);
            LongCiphertext raI_secret_a = ra_secret_a.multiply_plain(xbWI_b_plain, evaluator);
            raI_secret_a.rescale_to_next_inplace(evaluator);
            raI_secret_a.scale(scale);

            std::vector<double> temp_raI(inp_seq * d_k);
            auto temp_raI1 = matmul(ra_xa, WIb, inp_seq, d_module, d_k);
            auto temp_raI2 = matmul(input_b, ra_WIa, inp_seq, d_module, d_k);
            for (size_t i = 0; i < inp_seq * d_k; i++)
                temp_raI[i] = ra_xa_WIa[i] + temp_raI1[i] + temp_raI2[i];
            LongPlaintext temp_raI_plain(temp_raI, scale, party->slot_count, encoder);
            temp_raI_plain.mod_switch_to_inplace(raI_secret_a.parms_id(), evaluator);
            raI_secret_a.add_plain_inplace(temp_raI_plain, evaluator);
            return raI_secret_a;
        };
        // [r_aQ]_A
        LongCiphertext raQ_sec_a = cal_raI_A(input, WQ, ra_xa, ra_WQa, ra_xa_WQa, ra_secret_a,
                                             party, encoder, evaluator, scale, inp_seq, d_module, d_k);
        // [r_aK]_A
        LongCiphertext raK_sec_a = cal_raI_A(input, WK, ra_xa, ra_WKa, ra_xa_WKa, ra_secret_a,
                                             party, encoder, evaluator, scale, inp_seq, d_module, d_k);
        // [r_aV]_A
        LongCiphertext raV_sec_a = cal_raI_A(input, WV, ra_xa, ra_WVa, ra_xa_WVa, ra_secret_a,
                                             party, encoder, evaluator, scale, inp_seq, d_module, d_k);
        double rb1 = dist(gen), div_rb1 = 1 / rb1, rb1_square = rb1 * rb1;
        Plaintext div_rb1_plain_;
        encoder->encode(div_rb1, scale, div_rb1_plain_);
        Plaintext rb1_square_plain;
        encoder->encode(rb1_square, scale, rb1_square_plain);
        Ciphertext rb1_square_secret_b_;
        party->encryptor->encrypt(rb1_square_plain, rb1_square_secret_b_);
        LongCiphertext rb1_square_secret_b(rb1_square_secret_b_);
        LongPlaintext div_rb1_plain(div_rb1_plain_, party->slot_count);
        div_rb1_plain.mod_switch_to_inplace(raQ_sec_a.parms_id(), evaluator);
        raQ_sec_a.multiply_plain_inplace(div_rb1_plain, evaluator);
        raK_sec_a.multiply_plain_inplace(div_rb1_plain, evaluator);
        LongCiphertext::send(io_pack, &raQ_sec_a);
        LongCiphertext::send(io_pack, &raK_sec_a);
        LongCiphertext::send(io_pack, &rb1_square_secret_b);
        // send H2 = {raQ_sec_a, raK_sec_a, rb1_square_secret_b} to Alice

        /*
            Bob receive H3, and get Zb, [exp(Zc)]_a
            1. generate rb2, Ds, Rs, O randomly, which: sum_j O_ij = 0; Ds is column vector, Rs is row
               vector, they expand to a matrix
            2. compute [rb2*exp(Z) + O]_c = rb2*exp(Zb)*[exp(Za)]_C + O, Db*exp(Zb), Rb*[rcV]_c
            3. Z = r / (Db * Rb^T)
        */
        LongCiphertext Zb_secret_b, eZa_secret_a;
        LongCiphertext::recv(io_pack, &Zb_secret_b, context);
        LongCiphertext::recv(io_pack, &eZa_secret_a, context);
        LongPlaintext eZb_plain = Zb_secret_b.decrypt(party);
        std::vector<double> eZb = eZb_plain.decode(encoder);
        double rb2 = dist(gen);
        std::vector<double> Db(inp_seq);
        random_mat(Db);
        std::vector<double> O = zero_sum(inp_seq, inp_seq);
        for (size_t i = 0; i < inp_seq * inp_seq; i++)
        {
            eZb[i] = exp(eZb[i]) * rb2;
        }
        LongPlaintext rb2_expZb_plain(eZb, scale, party->slot_count, encoder);
        eZa_secret_a.multiply_plain_inplace(rb2_expZb_plain, evaluator);
        eZa_secret_a.rescale_to_next_inplace(evaluator);
        eZa_secret_a.scale(scale);
        LongPlaintext O_plain(O, scale, party->slot_count, encoder);
        O_plain.mod_switch_to_inplace(eZa_secret_a.parms_id(), evaluator);
        eZa_secret_a.add_plain_inplace(O_plain, evaluator);

        for (size_t i = 0; i < inp_seq * inp_seq; i++)
            eZb[i] = eZb[i] * Db[i / inp_seq] / rb2;
        std::vector<double> Rb(inp_seq * d_k);
        random_mat(Rb);
        for (i = 1; i < inp_seq; i++)
            for (j = 0; j < d_k; j++)
                Rb[i * d_k + j] = Rb[j];
        LongPlaintext Rb_plain(Rb, scale, party->slot_count, encoder);
        Rb_plain.mod_switch_to_inplace(raV_sec_a.parms_id(), evaluator);
        raV_sec_a.multiply_plain_inplace(Rb_plain, evaluator);

        for (i = 0; i < inp_seq; i++)
            for (j = 0; j < d_k; j++)
                Attn[i * d_k + j] = rb2 / (Db[i] * Rb[j]);
        LongPlaintext Attn_plain(Attn, scale, party->slot_count, encoder);
        LongCiphertext Attn_secret_b(Attn_plain, party);

        LongCiphertext::send(io_pack, &eZa_secret_a);
        send_mat(io_pack, &eZb);
        LongCiphertext::send(io_pack, &raV_sec_a);
        LongCiphertext::send(io_pack, &Attn_secret_b);
        // send H4 = {eZa_secret_a, eZb, raV_sec_a, [Attn]_s} to alice
    }
    std::cout << "Secure Attention done.\n";
    return output;
}

Multi_Head_Attention::Multi_Head_Attention(CKKSKey *party, SEALContext *context, IOPack *io_pack,
                                           size_t n_head_, size_t d_module, size_t d_k) : n_head(n_head_)
{
    attns = new Attention *[n_head];
    for (size_t i = 0; i < n_head; i++)
    {
        attns[i] = new Attention(party, context, io_pack, d_module, d_k, i);
    }
}

Multi_Head_Attention::~Multi_Head_Attention()
{
    for (size_t i = 0; i < n_head; i++)
    {
        delete attns[i];
    }
    delete[] attns;
}

LongCiphertext Multi_Head_Attention::forward(const std::vector<double> &input)
{
    LongCiphertext output = attns[0]->forward(input);
    for (size_t i = 1; i < n_head; i++)
    {
        LongCiphertext lct = attns[i]->forward(input);
        if (attns[i]->party->party == ALICE)
        {
            output.add_inplace(lct, attns[i]->evaluator);
        }
    }
    return output;
}