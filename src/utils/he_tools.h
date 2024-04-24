#ifndef FAST_HE_TOOLS_H__
#define FAST_HE_TOOLS_H__
#include <cassert>
#include <sstream>
#include <string>

#include <config.h>
#include <seal/seal.h>

#include "netio.h"

using namespace seal;

class length_error : public std::exception
{
    const char *message;

public:
    length_error(const char *msg) : message(msg) {}
    const char *what() const throw() override
    {
        return message;
    }
};

class CKKSKey
{
public:
    int party;
    size_t slot_count;
    SEALContext *context;
    KeyGenerator *keygen;
    Encryptor *encryptor;
    Decryptor *decryptor;
    PublicKey public_key;
    CKKSKey(int party_, SEALContext *context_, size_t slot_count_);
    ~CKKSKey();
};

class LongPlaintext
{
public:
    std::vector<Plaintext> plain_data;
    size_t len;
    size_t slot_count;
    LongPlaintext(size_t slot_count_) : slot_count(slot_count_) {}
    LongPlaintext(Plaintext pt, size_t slot_count_);
    LongPlaintext(std::vector<double> data, double scale, size_t slot_count_, CKKSEncoder *encoder);
    std::vector<double> decode(CKKSEncoder *encoder);

    inline void mod_switch_to_inplace(parms_id_type parms_id, Evaluator *evaluator)
    {
        for (size_t i = 0; i < plain_data.size(); i++)
        {
            evaluator->mod_switch_to_inplace(plain_data[i], parms_id);
        }
    }
};

class LongCiphertext
{
public:
    std::vector<Ciphertext> cipher_data;
    size_t len;
    LongCiphertext() {}
    LongCiphertext(Ciphertext ct);
    LongCiphertext(LongPlaintext lpt, CKKSKey *party);
    LongPlaintext decrypt(CKKSKey *party);
    void add_plain_inplace(LongPlaintext &lpt, Evaluator *evaluator);
    LongCiphertext add_plain(LongPlaintext &lpt, Evaluator *evaluator);
    void add_inplace(LongCiphertext &lct, Evaluator *evaluator);
    LongCiphertext add(LongCiphertext &lct, Evaluator *evaluator);
    void multiply_plain_inplace(LongPlaintext &lpt, Evaluator *evaluator);
    LongCiphertext multiply_plain(LongPlaintext &lpt, Evaluator *evaluator);
    static void send(IOPack *io_pack, LongCiphertext *lct);
    static void recv(IOPack *io_pack, LongCiphertext *lct, SEALContext *context);

    inline void rescale_to_next_inplace(Evaluator *evaluator)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            evaluator->rescale_to_next_inplace(cipher_data[i]);
        }
    }

    inline void scale(double scale_)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            cipher_data[i].scale() = scale_;
        }
    }

    inline parms_id_type parms_id()
    {
        return cipher_data[0].parms_id();
    }
};
#endif