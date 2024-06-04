#include <model.h>
#define TEST

// class SecureLayerNorm1
// {
//     BFVKey *alice;
//     BFVKey *bob;
//     BatchEncoder *encoder;
//     Evaluator *evaluator;

// public:
//     SecureLayerNorm1(BFVKey *alice_, BFVKey *bob_,
//                      BatchEncoder *encoder_, Evaluator *evaluator_) : alice(alice_),
//                                                                       bob(bob_),
//                                                                       encoder(encoder_),
//                                                                       evaluator(evaluator_) {}

//     void forward(BFVLongCiphertext &attn_s, const bfv_matrix &input_a, const bfv_matrix &input_b)
//     {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<> dist(-1, 1);
//         size_t i, j;

//         double ha1 = dist(gen), ha2 = dist(gen);
//         bfv_matrix ha1_xa(input_a.size());
//         for (i = 0; i < batch_size * d_module; i++)
//         {
//             ha1_xa[i] = ha1 * input_a[i];
//         }
//         BFVLongCiphertext ha2_div_ha1_secret_a(ha2 / ha1, alice, encoder);
//         BFVLongPlaintext ha2_plain(ha2, encoder);
//         BFVLongCiphertext ha2_secret_a(ha2_plain, alice);
//         BFVLongCiphertext attn_ha2_b = attn_s.multiply_plain(ha2_plain, evaluator);
//         // send H1 = {hc1_xc, hc2_div_hc1_secret_a, hc2_secret_a, attn_hc2_s} to bob

//         // bob receive H1, and get hc1_xc, hc2_div_hc1_secret_a, hc2_secret_a, attn_hc2
//         auto attn_ha2_plain = attn_ha2_b.decrypt(bob);
//         BFVLongPlaintext input_b_plain(input_b, encoder);
//         BFVLongCiphertext xha1_secret_a = ha2_secret_a.multiply_plain(input_b_plain, evaluator);
//         attn_ha2_plain.mod_switch_to_inplace(xha1_secret_a.parms_id(), evaluator);
//         xha1_secret_a.add_plain_inplace(attn_ha2_plain, evaluator);

//         BFVLongPlaintext ha1_xc_plain(ha1_xa, encoder);
//         ha2_div_ha1_secret_a.multiply_plain_inplace(ha1_xc_plain, evaluator);
//         ha2_div_ha1_secret_a.mod_switch_to_inplace(xha1_secret_a.parms_id(), evaluator);
//         xha1_secret_a.add_inplace(ha2_div_ha1_secret_a, evaluator);

//         double gb = dist(gen);
//         gb = 1;
//         BFVLongPlaintext gb_plain(gb, encoder);
//         gb_plain.mod_switch_to_inplace(xha1_secret_a.parms_id(), evaluator);
//         xha1_secret_a.multiply_plain_inplace(gb_plain, evaluator);
//         // send H2 = {xhc1_secret_a} to alice;

//         // alice receive H2, and get x * gb
//         auto xgb_plain = xha1_secret_a.decrypt(alice);
//         auto xgb = xgb_plain.decode(encoder);
//         for (i = 0; i < batch_size * d_module; i++)
//         {
//             xgb[i] /= ha2;
//         }
//         double ka = dist(gen);
//         auto mu_gb = mean(xgb, batch_size, d_module);
//         auto sigma_gb = standard_deviation(xgb, mu_gb, batch_size, d_module);
//         bfv_matrix div_sigma_gb(batch_size * d_module);
//         bfv_matrix tmp1(batch_size * d_module);
//         for (i = 0; i < batch_size; i++)
//         {
//             for (j = 0; j < d_module; j++)
//             {
//                 tmp1[i * d_module + j] = (xgb[i * d_module + j] - mu_gb[i]) * ka;
//                 div_sigma_gb[i * d_module + j] = 1 / (sigma_gb[i] * ka);
//             }
//         }
//         BFVLongPlaintext div_sigma_gb_plain(div_sigma_gb, encoder);
//         BFVLongCiphertext tmp2_secret_a(div_sigma_gb_plain, alice);
//         // send H3 = {tmp1, tmp2_secret_a} to bob

//         // bob receive H3
//         bfv_matrix gamma(batch_size * d_module);
//         bfv_matrix beta(batch_size * d_module);
//         random_mat(gamma);
//         random_mat(beta);
//         for (i = 0; i < batch_size * d_module; i++)
//         {
//             tmp1[i] *= gamma[i];
//         }
//         BFVLongPlaintext gamma_tmp1_plain(tmp1, encoder), beta_plain(beta, encoder);
//         BFVLongCiphertext ln_secret_a = tmp2_secret_a.multiply_plain(gamma_tmp1_plain, evaluator);
//         beta_plain.mod_switch_to_inplace(ln_secret_a.parms_id(), evaluator);
//         ln_secret_a.add_plain_inplace(beta_plain, evaluator);
//         std::cout << "Secure LayerNorm1 done.\n";

// #ifdef TEST
//         auto ln_plain = ln_secret_a.decrypt(alice);
//         auto ln = ln_plain.decode(encoder);

//         auto attn_plain = attn_s.decrypt(bob);
//         auto attn = attn_plain.decode(encoder);
//         for (i = 0; i < batch_size * d_module; i++)
//         {
//             attn[i] += (input_a[i] + input_b[i]);
//         }
//         auto mu = mean(attn, batch_size, d_module);
//         auto sigma = standard_deviation(attn, mu, batch_size, d_module);
//         for (i = 0; i < batch_size; i++)
//         {
//             for (j = 0; j < d_module; j++)
//             {
//                 attn[i * d_module + j] -= mu[i];
//                 attn[i * d_module + j] /= sigma[i];
//                 attn[i * d_module + j] *= gamma[i * d_module + j];
//                 attn[i * d_module + j] += beta[i * d_module + j];
//                 attn[i * d_module + j] -= ln[i * d_module + j];
//             }
//         }
//         std::cout << "error:"
//                   << "\n";
//         print_mat(attn, batch_size, d_module);
// #endif
//     }
// };

// int main()
// {
//     EncryptionParameters parms(scheme_type::ckks);
//     parms.set_poly_modulus_degree(poly_modulus_degree);
//     parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
//     SEALContext *context = new SEALContext(parms);
//     BatchEncoder *encoder = new BatchEncoder(*context);
//     Evaluator *evaluator = new Evaluator(*context);
//     BFVKey *alice = new BFVKey(1, context);
//     BFVKey *bob = new BFVKey(2, context);

//     bfv_matrix attn(batch_size * d_module), input_a(batch_size * d_module), input_b(batch_size * d_module);
//     random_mat(attn);
//     random_mat(input_a);
//     random_mat(input_b);
//     BFVLongPlaintext attn_plain(attn, encoder);
//     BFVLongCiphertext attn_secret_s(attn_plain, bob);

//     size_t i, j;
//     SecureLayerNorm1 *sec_ln1 = new SecureLayerNorm1(alice, bob, encoder, evaluator);
//     sec_ln1->forward(attn_secret_s, input_a, input_b);

//     delete sec_ln1;
//     delete context;
//     delete encoder;
//     delete evaluator;
//     delete alice;
//     delete bob;
// }

int main()
{
    uint64_t *ret_int64 = new uint64_t[10];
    sci::PRG128 prg;
    prg.random_data(ret_int64, 10 * sizeof(uint64_t));
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << ret_int64[i] << " ";
    }
}