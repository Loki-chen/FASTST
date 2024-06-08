#include <model.h>
#include "ezpc_scilib/ezpc_utils.h"
#define TEST

class SecureLayerNorm1
{
    BFVKey *alice;
    BFVKey *bob;
    BatchEncoder *encoder;
    Evaluator *evaluator;

    sci::IOPack *iopack;
    sci::OTPack *otpack;

public:
    SecureLayerNorm1(BFVKey *alice_, BFVKey *bob_,
                     BatchEncoder *encoder_, Evaluator *evaluator_, sci::IOPack *iopack_, sci::OTPack *otpack_) : alice(alice_),
                                                                                                                  bob(bob_),
                                                                                                                  encoder(encoder_),
                                                                                                                  evaluator(evaluator_),
                                                                                                                  iopack(iopack_),
                                                                                                                  otpack(otpack_) {}
    void forward(BFVLongCiphertext &att_s, const bfv_matrix &input_a, const bfv_matrix &input_b)
    {

        sci::PRG128 prg;
        size_t i, j;
        uint64_t *ha1 = new uint64_t[1];
        uint64_t *ha2 = new uint64_t[1];
        prg.random_data(ha1, sizeof(uint64_t));
        prg.random_data(ha2, sizeof(uint64_t));
        FixOp *fix_alice = new FixOp(sci::ALICE, iopack, otpack);
        FixOp *fix_bob = new FixOp(sci::BOB, iopack, otpack);
        FixOp *fix_public = new FixOp(sci::PUBLIC, iopack, otpack);

        uint64_t *intput_data_a = new uint64_t[input_a.size()];
        uint64_t *intput_data_b = new uint64_t[input_b.size()];

        for (size_t i = 0; i < input_a.size(); i++)
        {
            intput_data_a[i] = input_a[i];
        }
        for (size_t i = 0; i < input_b.size(); i++)
        {
            intput_data_b[i] = input_b[i];
        }

        FixArray fix_input_a = fix_alice->input(sci::ALICE, input_a.size(), intput_data_a, true, 64, 13);
        FixArray fix_input_b = fix_bob->input(sci::BOB, input_b.size(), intput_data_b, true, 64, 13);

        FixArray ha1_xa = fix_alice->mul(fix_input_a, ha1[0]);
    }
};

int main()
{
    std::cout << default_prime_mod.at(29) << "\n";
    BFVparm *bfv_alice_parm = new BFVparm(sci::ALICE, 8192, {54, 54, 55, 55}, default_prime_mod.at(29));
    BFVparm *bfv_bob_parm = new BFVparm(sci::BOB, 8192, {54, 54, 55, 55}, default_prime_mod.at(29));

    BFVKey *alice = new BFVKey(bfv_alice_parm->party, bfv_alice_parm->context);
    BFVKey *bob = new BFVKey(bfv_bob_parm->party, bfv_bob_parm->context);

    BatchEncoder *encoder = new BatchEncoder(*bfv_bob_parm->context);

    Evaluator *evaluator = new Evaluator(*bfv_bob_parm->context);
    bfv_matrix attn(batch_size * d_module),
        input_a(batch_size * d_module), input_b(batch_size * d_module);
    random_bfv_mat(attn);
    random_bfv_mat(input_a);
    random_bfv_mat(input_b);

    BFVLongPlaintext attn_plain(bfv_bob_parm, attn, encoder); ///////?///////// nmsl
    // BFVLongCiphertext attn_secret_s(attn_plain, bob);
    // size_t i, j;
    // sci::IOPack *iopack;
    // sci::OTPack *otpack;
    // SecureLayerNorm1 *sec_ln1 = new SecureLayerNorm1(alice, bob, encoder, evaluator, iopack, otpack);
    // sec_ln1->forward(attn_secret_s, input_a, input_b);
}