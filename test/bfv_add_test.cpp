#include <model.h>
#define TEST
int main()
{

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(CoeffModulus::Create(bfv_poly_modulus_degree, bfv_coeff_bit_sizes));
    parms.set_plain_modulus(bfv_plain_mod);

    SEALContext *context = new SEALContext(parms);
    BatchEncoder *encoder = new BatchEncoder(*context);
    Evaluator *evaluator = new Evaluator(*context);
    BFVKey *alice = new BFVKey(1, context);
    BFVKey *bob = new BFVKey(2, context);

    vector<uint64_t> plain_vec;
    for (size_t i = 0; i < encoder->slot_count(); i++)
    {
        plain_vec.push_back(i);
    }

    BFVLongPlaintext pt(plain_vec, encoder);
    auto res = pt.decode(encoder);
    for (size_t i = 0; i < 10; i++)
    {
        cout << res[i] << " ";
    }
    cout << "" << std::endl;
}