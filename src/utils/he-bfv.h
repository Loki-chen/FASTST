#ifndef FAST_HE_BFV_TOOLS_H__
#define FAST_HE_BFV_TOOLS_H__
#pragma once
#include <cassert>
#include <sstream>
#include <string>

#include <party.h>
#include <seal/seal.h>

#include "io.h"

using std::map;
using std::string;
using std::vector;
using namespace seal;

typedef vector<uint64_t> bfv_matrix;

const map<int32_t, uint64_t> default_prime_mod{
    {25, 33832961},
    {28, 268582913},
    {29, 536903681},
    {30, 1073872897},
    {31, 2146959361},
    {32, 4293918721},
    {33, 8585084929},
    {34, 17171218433},
    {35, 34359214081},
    {36, 68686184449},
    {37, 137352314881},
    {38, 274824036353},
    {39, 549753716737},
    {40, 1099480956929},
    {41, 2198100901889},
};

const size_t bfv_poly_modulus_degree = 8192;
const size_t bfv_slot_count = bfv_poly_modulus_degree;
const vector<int> bfv_coeff_bit_sizes = {54, 54, 55, 55};
const int32_t bitlength = 29;
const uint64_t bfv_plain_mod = default_prime_mod.at(bitlength);
class bfv_lenth_error : public std::exception
{
    const char *message;

public:
    bfv_lenth_error(const char *msg) : message(msg) {}
    const char *what() const throw() override
    {
        return message;
    }
};

class BFVKey
{
public:
    int party;
    SEALContext *context;
    KeyGenerator *keygen;
    Encryptor *encryptor;
    Decryptor *decryptor;
    PublicKey public_key;
    RelinKeys relin_keys;

    BFVKey(int party_, SEALContext *context_);
    ~BFVKey();
};

class BFVLongPlaintext
{
public:
    vector<Plaintext> plain_data;
    size_t len;
    BFVLongPlaintext() {}
    BFVLongPlaintext(const Plaintext &pt);
    BFVLongPlaintext(uint64_t data, BatchEncoder *encoder); // TODO: len=1
    BFVLongPlaintext(bfv_matrix data, BatchEncoder *encoder);
    bfv_matrix decode(BatchEncoder *encoder) const;
};

class BFVLongCiphertext
{
public:
    vector<Ciphertext> cipher_data;
    size_t len;
    BFVLongCiphertext() {}
    BFVLongCiphertext(const Ciphertext &ct);
    BFVLongCiphertext(uint64_t data, BFVKey *party, BatchEncoder *encoder); // TODO: len =1
    BFVLongCiphertext(const BFVLongPlaintext &lpt, BFVKey *party);
    BFVLongPlaintext decrypt(BFVKey *party) const;

    void add_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator);
    BFVLongCiphertext add_plain(BFVLongPlaintext &lpt, Evaluator *evaluator) const;
    void add_inplace(BFVLongCiphertext &lct, Evaluator *evaluator);
    BFVLongCiphertext add(BFVLongCiphertext &lct, Evaluator *evaluator) const;
    void multiply_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator, RelinKeys *relin_keys = nullptr);
    BFVLongCiphertext multiply_plain(BFVLongPlaintext &lpt, Evaluator *evaluator, RelinKeys *relin_keys = nullptr) const;
    static void send(IOPack *io_pack, BFVLongCiphertext *lct);
    static void recv(IOPack *io_pack, BFVLongCiphertext *lct, SEALContext *context);

    inline const parms_id_type parms_id() const noexcept
    {
        return cipher_data[0].parms_id();
    }

    inline void print_parameters(std::shared_ptr<seal::SEALContext> context)
    {
        // Verify parameters
        if (!context)
        {
            throw std::invalid_argument("context is not set");
        }
        auto &context_data = *context->key_context_data();

        /*
        Which scheme are we using?
        */
        std::string scheme_name;
        switch (context_data.parms().scheme())
        {
        case seal::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case seal::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
        }
        std::cout << "/" << std::endl;
        std::cout << "| Encryption parameters :" << std::endl;
        std::cout << "|   scheme: " << scheme_name << std::endl;
        std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

        /*
        Print the size of the true (product) coefficient modulus.
        */
        std::cout << "|   coeff_modulus size: ";
        std::cout << context_data.total_coeff_modulus_bit_count() << " (";
        auto coeff_modulus = context_data.parms().coeff_modulus();
        std::size_t coeff_mod_count = coeff_modulus.size();
        for (std::size_t i = 0; i < coeff_mod_count - 1; i++)
        {
            std::cout << coeff_modulus[i].bit_count() << " + ";
        }
        std::cout << coeff_modulus.back().bit_count();
        std::cout << ") bits" << std::endl;

        /*
        For the BFV scheme print the plain_modulus parameter.
        */
        if (context_data.parms().scheme() == seal::scheme_type::bfv)
        {
            std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
        }

        std::cout << "\\" << std::endl;
    }
};
#endif // FAST_HE_BFV_TOOLS_H__