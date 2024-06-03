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


    // Batch test: matrix len <= slot count
    vector<uint64_t> plain_vec;
    for (size_t i = 0; i < encoder->slot_count(); i++)
    {
        plain_vec.push_back(i);
    }
    BFVLongPlaintext pt(plain_vec, encoder);  //encode
    auto res = pt.decode(encoder);  //decode
    BFVLongCiphertext ct(pt, alice);  // encrypt
    BFVLongPlaintext pt_dec = ct.decrypt(alice); // decrypt
    bfv_matrix res_dec = pt_dec.decode(encoder); 
    for (size_t i = 0; i < 10; i++)
    {
        cout << res[i] << " ";
        cout << res_dec[i] << " ";
    }

    // How to enc and dec for a single data?  
    // use a matrix with len = n. we just use the first slot.
    uint64_t sing_data = 6 ;
    BFVLongPlaintext single_pt(sing_data, encoder);
    auto sing_res = single_pt.decode(encoder);

    BFVLongCiphertext sing_ct(single_pt, bob);
    BFVLongPlaintext sing_ct_dec = sing_ct.decrypt(bob);
    bfv_matrix sing_ct_plain = sing_ct_dec.decode(encoder);
    cout << sing_res[0] << " ";
    cout << sing_ct_plain[0] << " ";



    cout << "" << std::endl;
}