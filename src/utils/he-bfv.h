#ifndef FAST_HE_BFV_TOOLS_H__
#define FAST_HE_BFV_TOOLS_H__

#include <cassert>
#include <seal/seal.h>
#include <sstream>
#include <string>

#include <Utils/emp-tool.h> // prg.h & io & arg
#include <seal/seal.h>

using std::map;
using std::string;
using std::vector;
using namespace seal;

const map<int32_t, uint64_t> default_prime_mod{
    {13,        557057},
    {16,       5875273},
    {25,      33832961},
    {28,     268582913},
    {29,     536903681},
    {30,    1073872897},
    {31,    2146959361},
    {32,    4293918721},
    {33,    8585084929},
    {34,   17171218433},
    {35,   34359214081},
    {36,   68686184449},
    {37,  137352314881},
    {38,  274824036353},
    {39,  549753716737},
    {40, 1099480956929},
    {41, 2198100901889},
};

// const size_t bfv_poly_modulus_degree = 8192;
// const size_t bfv_slot_count = bfv_poly_modulus_degree;
// const vector<int> bfv_coeff_bit_sizes = {54, 54, 55, 55};
// const int32_t bitlength = 29;
// const uint64_t bfv_plain_mod = default_prime_mod.at(bitlength);

void print_parameters(std::shared_ptr<seal::SEALContext> context);

class BFVParm {
public:
    size_t poly_modulus_degree;
    size_t slot_count;
    vector<int> coeff_bit_sizes;
    // int32_t bitlength;
    uint64_t plain_mod;
    SEALContext *context;
    BatchEncoder *encoder;
    Evaluator *evaluator;
    BFVParm(size_t poly_modulus_degree, vector<int> coeff_bit_sizes, uint64_t plain_mod);
    ~BFVParm();
};

class bfv_lenth_error : public std::exception {
    const char *message;

public:
    bfv_lenth_error(const char *msg) : message(msg) {}
    const char *what() const throw() override { return message; }
};

class BFVKey {
public:
    int party;
    BFVParm *parm;
    Encryptor *encryptor;
    Decryptor *decryptor;
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;

    BFVKey(int party_, BFVParm *parm_);
    ~BFVKey();
};

class BFVLongPlaintext {
public:
    vector<Plaintext> plain_data;
    size_t len;
    BFVLongPlaintext() {}
    BFVLongPlaintext(const Plaintext &pt);

    BFVLongPlaintext(BFVParm *contex, uint64_t data); // TODO: len=1
    BFVLongPlaintext(BFVParm *contex, vector<uint64_t> data);
    BFVLongPlaintext(BFVParm *contex, uint64_t *data, size_t len);
    vector<uint64_t> decode_uint(BFVParm *contex) const;

    // for int64_t data
    BFVLongPlaintext(BFVParm *contex, int64_t data); // TODO: len=1
    BFVLongPlaintext(BFVParm *contex, vector<int64_t> data);
    BFVLongPlaintext(BFVParm *contex, int64_t *data, size_t len);
    vector<int64_t> decode_int(BFVParm *contex) const;

    inline void mod_switch_to_inplace(parms_id_type parms_id, Evaluator *evaluator) {
        size_t p_d_size = plain_data.size();
        for (size_t i = 0; i < p_d_size; i++) {
            evaluator->mod_switch_to_inplace(plain_data[i], parms_id);
        }
    }
};

class BFVLongCiphertext {
public:
    vector<Ciphertext> cipher_data;
    size_t len;
    BFVLongCiphertext() {}
    BFVLongCiphertext(const Ciphertext &ct);

    BFVLongCiphertext(BFVParm *contex, uint64_t data, BFVKey *party); // TODO: len =1
    BFVLongCiphertext(BFVParm *contex, uint64_t *data, size_t len, BFVKey *party);

    BFVLongCiphertext(BFVParm *contex, int64_t data, BFVKey *party); // TODO: len =1
    BFVLongCiphertext(BFVParm *contex, int64_t *data, size_t len, BFVKey *party);

    BFVLongCiphertext(const BFVLongPlaintext &lpt, BFVKey *party);
    BFVLongPlaintext decrypt(BFVKey *party) const;

    void add_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator);
    BFVLongCiphertext add_plain(BFVLongPlaintext &lpt, Evaluator *evaluator) const;
    void add_inplace(BFVLongCiphertext &lct, Evaluator *evaluator);
    BFVLongCiphertext add(BFVLongCiphertext &lct, Evaluator *evaluator) const;
    void sub_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator);
    BFVLongCiphertext sub_plain(BFVLongPlaintext &lpt, Evaluator *evaluator) const;
    void sub_inplace(BFVLongCiphertext &lct, Evaluator *evaluator);
    BFVLongCiphertext sub(BFVLongCiphertext &lct, Evaluator *evaluator) const;
    void multiply_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator);
    BFVLongCiphertext multiply_plain(BFVLongPlaintext &lpt, Evaluator *evaluator) const;
    void multiply_inplace(BFVLongCiphertext &lct, Evaluator *evaluator);
    BFVLongCiphertext multiply(BFVLongCiphertext &lct, Evaluator *evaluator) const;
    static void send(sci::NetIO *io, BFVLongCiphertext *lct);
    static void recv(sci::NetIO *io, BFVLongCiphertext *lct, SEALContext *context);

    inline void negate_inplace(Evaluator *evaluator) {
        for (auto &ct : cipher_data) {
            evaluator->negate_inplace(ct);
        }
    }

    inline BFVLongCiphertext negate(Evaluator *evaluator) const {
        size_t c_d_size = cipher_data.size();
        BFVLongCiphertext ret;
        ret.cipher_data = vector<Ciphertext>(c_d_size);
        ret.len = len;
        for (size_t i = 0; i < c_d_size; i++) {
            evaluator->negate(cipher_data[i], ret.cipher_data[i]);
        }
        return ret;
    }

    inline void square_inplace(Evaluator *evaluator) {
        size_t c_d_size = cipher_data.size();
        for (size_t i = 0; i < c_d_size; i++) {
            evaluator->square_inplace(cipher_data[i]);
        }
    }

    inline BFVLongCiphertext square(Evaluator *evaluator) const {
        size_t size = cipher_data.size();
        BFVLongCiphertext ret;
        ret.cipher_data.resize(size);
        ret.len = len;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            evaluator->square(cipher_data[i], ret.cipher_data[i]);
        }
        return ret;
    }

    inline void mod_switch_to_inplace(parms_id_type parms_id, Evaluator *evaluator) {
        size_t size = cipher_data.size();
        for (size_t i = 0; i < size; i++) {
            evaluator->mod_switch_to_inplace(cipher_data[i], parms_id);
        }
    }

    inline void mod_switch_to_next_inplace(Evaluator *evaluator) {
        size_t size = cipher_data.size();
        for (size_t i = 0; i < size; i++) {
            evaluator->mod_switch_to_next_inplace(cipher_data[i]);
        }
    }

    inline void relinearize_inplace(Evaluator *evaluator, const RelinKeys &relin_keys) {
        size_t size = cipher_data.size();
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            evaluator->relinearize_inplace(cipher_data[i], relin_keys);
        }
    }

    inline const parms_id_type parms_id() const noexcept { return cipher_data[0].parms_id(); }
};

#endif
// FAST_HE_BFV_TOOLS_H__