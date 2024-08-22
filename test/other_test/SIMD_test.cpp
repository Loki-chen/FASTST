#include <utils.h>

int64_t mod_inverse(int64_t a, int64_t m) {
    int64_t m0 = m, x0 = 0, x1 = 1;

    while (a > 1) {
        int64_t q = a / m;
        int64_t temp = m;
        m = a % m;
        a = temp;
        int64_t temp_x = x0;
        x0 = x1 - q * x0;
        x1 = temp_x;
    }

    return x1 < 0 ? x1 + m0 : x1;
}

int main() {
    std::stringstream os;
    BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
    BFVKey *party = new BFVKey(1, bfv_parm);
    vector<uint64_t> data(bfv_parm->slot_count);
    vector<uint64_t> inv_data(bfv_parm->slot_count);
    for (int i = 0; i < bfv_parm->slot_count; i++) {
        data[i] = (i + 1) * 1ull << 12;
        inv_data[i] = mod_inverse(data[i], default_prime_mod.at(29));
    }
    Plaintext pt, inv_pt;
    bfv_parm->encoder->encode(data, pt);
    bfv_parm->encoder->encode(inv_data, inv_pt);
    Ciphertext ct;
    party->encryptor->encrypt(pt, ct);
    bfv_parm->evaluator->multiply_plain_inplace(ct, inv_pt);
    party->decryptor->decrypt(ct, pt);
    bfv_parm->encoder->decode(pt, data);
    for (int i = 0; i < bfv_parm->slot_count; i++) {
        if (data[i] != 1) {
            std::cout << "error in: " << i << "\n";
            return 0;
        }
    }
    std::cout << "test pass\n";
}