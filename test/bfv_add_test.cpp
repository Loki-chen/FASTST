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

    // for (size_t i = 0; i < encoder->slot_count(); i++)
    // {
    //     std::cout << s[i] << " ";
    // }

    // sci::PRG128 prg;
    // uint64_t *secret_share = new uint64_t[2 * 2];
    // prg.random_mod_p<uint64_t>(secret_share, 4, 536903681);
    // for (size_t i = 0; i < 4; i++)
    // {
    //     std::cout << secret_share[i] << " ";
    // }

    // bfv_matrix mat(2 * 2);

    // size_t size = mat.size();
    // uint64_t *rand_mod_P_num = new uint64_t[size];
    // prg.random_mod_p<uint64_t>(rand_mod_P_num, size, 536903681);
    // for (size_t i = 0; i < size; i++)
    // {
    //     mat[i] = rand_mod_P_num[i];
    // }
    // for (size_t i = 0; i < 4; i++)
    // {
    //     std::cout << mat[i] << " ";
    // }
    sci::PRG128 prg;
    // size_t dim = 1;
    // uint64_t *ha1 = new uint64_t[dim];
    // prg2.random_mod_p<uint64_t>(ha1, dim * sizeof(uint64_t), 536903681);
    // for (size_t i = 0; i < dim; i++)
    // {
    //     std::cout << " data: " << ha1[i] << " !";
    // }

    bfv_matrix input(batch_size * d_module);
//     uint64_t bfv_plain_mod1 = 536903681;
#pragma omp parallel for
    for (size_t i = 0; i < batch_size * d_module; i++)
    {
        input[i] = i * 1025 % bfv_plain_mod;
    }

    //     uint64_t *ha11 = new uint64_t[16];
    //     prg.random_mod_p<uint64_t>(ha11, 16, bfv_plain_mod1);

    //     uint64_t *ha2 = new uint64_t[16];
    //     prg.random_mod_p<uint64_t>(ha11, 16, bfv_plain_mod1);

    //     bfv_matrix ha11_xa(input.size());
    // #pragma omp parallel for
    //     for (size_t i = 0; i < 16; i++)
    //     {
    //         ha11_xa[i] = (ha11[i] * input[i]) % bfv_plain_mod1;
    //     }

    uint64_t *ha1 = new uint64_t[batch_size * d_module];
    uint64_t *ha2 = new uint64_t[batch_size * d_module];
    prg.random_mod_p<uint64_t>(ha1, batch_size * d_module, bfv_plain_mod);
    prg.random_mod_p<uint64_t>(ha2, batch_size * d_module, bfv_plain_mod);

    bfv_matrix ha1_xa(input.size());

    for (size_t i = 0; i < batch_size * d_module; i++)
    {
        ha1_xa[i] = ModMult(ha1[i], input[i], bfv_plain_mod);
    }
    for (size_t i = 0; i < 16; i++)
    {
        std::cout << ha1_xa[i] << " ";
    }
    std::cout << "\n";
}