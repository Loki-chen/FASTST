#include "utils.h"
int main()
{
    vector<int64_t> a = {41235};
    vector<int64_t> b = {-13020}; // change it to 13021
    uint64_t mask = 1ULL << 16;
    // std::cout << "true result: " << double(a[0] * b[0]) / double(mask * mask) << "\n";
    uint64_t ture_res = a[0] * b[0];
    std::cout << "true result: " << a[0] * b[0] << "\n";

    BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(31));
    BFVKey *party = new BFVKey(1, bfv_parm);
    Plaintext pta, ptb, ptaa;
    bfv_parm->encoder->encode(a, pta), bfv_parm->encoder->encode(b, ptb);
    Ciphertext cta;
    party->encryptor->encrypt(pta, cta);
    bfv_parm->evaluator->multiply_plain_inplace(cta, ptb);
    party->decryptor->decrypt(cta, ptaa);
    vector<int64_t> res(1);
    bfv_parm->encoder->decode(ptaa, res);
    // std::cout << "result: " << double(res[0]) / double(default_prime_mod.at(29)) << "\n";
    std::cout << "result: " << res[0] << "\n";
}