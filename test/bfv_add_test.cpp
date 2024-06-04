#include <model.h>
#define TEST

int main()
{

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(bfv_poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(bfv_poly_modulus_degree, bfv_coeff_bit_sizes));
    parms.set_plain_modulus(bfv_plain_mod);

    SEALContext *context = new SEALContext(parms);
    BatchEncoder *encoder = new BatchEncoder(*context);
    Evaluator *evaluator = new Evaluator(*context);
    BFVKey *alice = new BFVKey(1, context);
    BFVKey *bob = new BFVKey(2, context);

    bfv_matrix plain_vec;
    for (size_t i = 0; i < encoder->slot_count(); i++)
    {
        plain_vec.push_back(i);
    }

    BFVLongPlaintext pt(plain_vec, encoder);
    BFVLongCiphertext ct(pt, alice);
    BFVLongCiphertext res = ct.multiply_plain(pt, evaluator);
    BFVLongPlaintext dec_res = res.decrypt(alice);
    bfv_matrix s = dec_res.decode(encoder);

    for (size_t i = 0; i < encoder->slot_count(); i++)
    {
        std::cout << s[i] << " ";
    }
}