#include <utils.h>

int main() {
    std::stringstream os;
    BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
    BFVKey *party = new BFVKey(1, bfv_parm);
    vector<uint64_t> data(bfv_parm->slot_count);
    data[0] = default_prime_mod.at(29) - 1;
    Plaintext pt(data);
    // std::cout << pt.to_string() << "\n";
    Ciphertext ct;
    party->encryptor->encrypt(pt, ct);
    bfv_parm->evaluator->square_inplace(ct);
    party->decryptor->decrypt(ct, pt);
    auto ret = pt.data();
    std::cout << ret[0] << "\n";
}