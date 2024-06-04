#include "he-bfv.h"

BFVKey::BFVKey(int party_, SEALContext *context_) : party(party_), context(context_)
{
    assert(party == ALICE || party == BOB);

    keygen = new KeyGenerator(*context);
    keygen->create_public_key(public_key);
    keygen->create_relin_keys(relin_keys);
    encryptor = new Encryptor(*context, public_key);
    decryptor = new Decryptor(*context, keygen->secret_key());
}

BFVKey::~BFVKey()
{
    delete keygen;
    delete encryptor;
    delete decryptor;
}

BFVLongPlaintext::BFVLongPlaintext(const Plaintext &pt)
{
    len = 1;
    plain_data.push_back(pt);
}

BFVLongPlaintext::BFVLongPlaintext(uint64_t data, BatchEncoder *encoder)
{
    // TODO value len =1, use the BFV batchencoder to encode the palaintext
    len = 1;
    Plaintext pt;
    vector<uint64_t> temp = {data};
    encoder->encode(temp, pt);
    plain_data.push_back(pt);
}

BFVLongPlaintext::BFVLongPlaintext(bfv_matrix data, BatchEncoder *encoder)
{
    len = data.size();
    size_t bfv_slot = bfv_slot_count;
    size_t count = len / bfv_slot;
    if (len % bfv_slot)
    {
        count++;
    }
    size_t i, j;
    if (bfv_slot >= len)
    {
        Plaintext pt;
        encoder->encode(data, pt);
        plain_data.push_back(pt);
    }
    else
    {
        bfv_matrix::iterator curPtr = data.begin(), endPtr = data.end(), end;
        while (curPtr < endPtr)
        {
            end = endPtr - curPtr > bfv_slot ? bfv_slot + curPtr : endPtr;
            bfv_slot = endPtr - curPtr > bfv_slot ? bfv_slot : endPtr - curPtr;
            bfv_matrix temp(curPtr, end);
            Plaintext pt;
            encoder->encode(temp, pt);
            plain_data.push_back(pt);
            curPtr += bfv_slot;
        }
    }
}

bfv_matrix BFVLongPlaintext::decode(BatchEncoder *encoder) const
{
    bfv_matrix data(len);
    size_t size = plain_data.size();
    for (size_t i = 0; i < size; i++)
    {
        bfv_matrix temp;
        encoder->decode(plain_data[i], temp);
        if (i < size - 1)
        {
            copy(temp.begin(), temp.end(), data.begin() + i * bfv_slot_count);
        }
        else
        {
            size_t tail_len = len % bfv_slot_count;
            tail_len = tail_len ? tail_len : bfv_slot_count;
            copy(temp.begin(), temp.begin() + tail_len + 1, data.begin() + i * bfv_slot_count);
        }
    }
    return data;
}

BFVLongCiphertext::BFVLongCiphertext(const Ciphertext &ct)
{
    len = 1;
    cipher_data.push_back(ct);
}

BFVLongCiphertext::BFVLongCiphertext(uint64_t data, BFVKey *party, BatchEncoder *encoder)
{
    // TODO:
    len = 1;
    Plaintext pt;
    vector<uint64_t> temp = {data};
    encoder->encode(temp, pt);
    Ciphertext ct;
    party->encryptor->encrypt(pt, ct);
    cipher_data.push_back(ct);
}

BFVLongCiphertext::BFVLongCiphertext(const BFVLongPlaintext &lpt, BFVKey *party)
{
    len = lpt.len;
    for (Plaintext pt : lpt.plain_data)
    {
        Ciphertext ct;
        party->encryptor->encrypt(pt, ct);
        cipher_data.push_back(ct);
    }
}

BFVLongPlaintext BFVLongCiphertext::decrypt(BFVKey *party) const
{
    BFVLongPlaintext lpt;
    lpt.len = len;
    for (Ciphertext ct : cipher_data)
    {
        Plaintext pt;
        party->decryptor->decrypt(ct, pt);

        lpt.plain_data.push_back(pt);
    }
    return lpt;
}

void BFVLongCiphertext::add_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator)
{
    if (len == 1) // cipher text len =1
    {
        len = lpt.len;
        Ciphertext ct(cipher_data[0]);
        cipher_data.pop_back();
        for (Plaintext pt : lpt.plain_data)
        {
            Ciphertext ctemp;
            evaluator->add_plain(ct, pt, ctemp);
            cipher_data.push_back(ctemp);
        }
    }
    else if (lpt.len == 1)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
            evaluator->add_plain_inplace(cipher_data[i], lpt.plain_data[0]);
    }
    else if (len == lpt.len)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
            evaluator->add_plain_inplace(cipher_data[i], lpt.plain_data[i]);
    }
    else
    {
        char buf[100];
        sprintf(buf, "Length of BFVLongCiphertext(%ld) and BFVLongPlaintext(%ld) mismatch", len, lpt.len);
        throw bfv_lenth_error(buf);
    }
}

BFVLongCiphertext BFVLongCiphertext::add_plain(BFVLongPlaintext &lpt, Evaluator *evaluator) const
{
    {
        BFVLongCiphertext lct;
        lct.len = 0;
        if (len == 1)
        {
            lct.len = lpt.len;
            for (size_t i = 0; i < lpt.plain_data.size(); i++)
            {
                Ciphertext ct;
                evaluator->add_plain(cipher_data[0], lpt.plain_data[i], ct);
                lct.cipher_data.push_back(ct);
            }
        }
        else if (lpt.len == 1)
        {
            lct.len = len;
            for (size_t i = 0; i < cipher_data.size(); i++)
            {
                Ciphertext ct;
                evaluator->add_plain(cipher_data[i], lpt.plain_data[0], ct);
                lct.cipher_data.push_back(ct);
            }
        }
        else if (len == lpt.len)
        {
            lct.len = len;
            for (size_t i = 0; i < cipher_data.size(); i++)
            {
                Ciphertext ct;
                evaluator->add_plain(cipher_data[i], lpt.plain_data[i], ct);
                lct.cipher_data.push_back(ct);
            }
        }
        else
        {
            char buf[100];
            sprintf(buf, "Length of BFVLongCiphertext(%ld) and LongPlaintext(%ld) mismatch", len, lpt.len);
            throw bfv_lenth_error(buf);
        }
        return lct;
    }
}

void BFVLongCiphertext::add_inplace(BFVLongCiphertext &lct, Evaluator *evaluator)
{
    if (len == 1)
    {
        len = lct.len;
        Ciphertext ct(cipher_data[0]);
        cipher_data.pop_back();
        for (Ciphertext cct : lct.cipher_data)
        {
            Ciphertext ctemp;
            evaluator->add(ct, cct, ctemp);
            cipher_data.push_back(ctemp);
        }
    }
    else if (lct.len == 1)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
            evaluator->add_inplace(cipher_data[i], lct.cipher_data[0]);
    }
    else if (len == lct.len)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
            evaluator->add_inplace(cipher_data[i], lct.cipher_data[i]);
    }
    else
    {
        char buf[100];
        sprintf(buf, "Length of BFVLongCiphertext(%ld) and BFVLongCiphertext(%ld) mismatch", len, lct.len);
        throw bfv_lenth_error(buf);
    }
}

BFVLongCiphertext BFVLongCiphertext::add(BFVLongCiphertext &lct, Evaluator *evaluator) const
{
    BFVLongCiphertext lcct;
    lcct.len = 0;
    if (len == 1)
    {
        lcct.len = lct.len;
        for (size_t i = 0; i < lct.cipher_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->add(cipher_data[0], lct.cipher_data[i], ct);
            lcct.cipher_data.push_back(ct);
        }
    }
    else if (lct.len == 1)
    {
        lcct.len = len;
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->add(cipher_data[i], lct.cipher_data[0], ct);
            lcct.cipher_data.push_back(ct);
        }
    }
    else if (len == lct.len)
    {
        lcct.len = len;
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->add(cipher_data[i], lct.cipher_data[i], ct);
            lcct.cipher_data.push_back(ct);
        }
    }
    else
    {
        char buf[100];
        sprintf(buf, "Length of BFVLongCiphertext(%ld) and BFVLongCiphertext(%ld) mismatch", len, lct.len);
        throw bfv_lenth_error(buf);
    }
    return lcct;
}

void BFVLongCiphertext::multiply_plain_inplace(BFVLongPlaintext &lpt, Evaluator *evaluator, RelinKeys *relin_keys)
{

    if (len == 1)
    {
        len = lpt.len;

        Ciphertext ct(cipher_data[0]);
        cipher_data.pop_back();
        for (Plaintext pt : lpt.plain_data)
        {
            Ciphertext ctemp;

            evaluator->multiply_plain(ct, pt, ctemp);

            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(ctemp, *relin_keys);
            }

            cipher_data.push_back(ctemp);
        }
    }
    else if (lpt.len == 1)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            evaluator->multiply_plain_inplace(cipher_data[i], lpt.plain_data[0]);
            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(cipher_data[i], *relin_keys);
            }
        }
    }
    else if (len == lpt.len)
    {
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            evaluator->multiply_plain_inplace(cipher_data[i], lpt.plain_data[i]);
            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(cipher_data[i], *relin_keys);
            }
        }
    }
    else
    {
        char buf[100];
        sprintf(buf, "Length of LongCiphertext(%ld) and LongPlaintext(%ld) mismatch", len, lpt.len);
        throw bfv_lenth_error(buf);
    }
}

BFVLongCiphertext BFVLongCiphertext::multiply_plain(BFVLongPlaintext &lpt, Evaluator *evaluator, RelinKeys *relin_keys) const
{
    BFVLongCiphertext lct;
    lct.len = 0;
    if (len == 1)
    {
        lct.len = lpt.len;
        for (size_t i = 0; i < lpt.plain_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->multiply_plain(cipher_data[0], lpt.plain_data[i], ct);
            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(ct, *relin_keys);
            }
            lct.cipher_data.push_back(ct);
        }
    }
    else if (lpt.len == 1)
    {
        lct.len = len;
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->multiply_plain(cipher_data[i], lpt.plain_data[0], ct);
            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(ct, *relin_keys);
            }

            lct.cipher_data.push_back(ct);
        }
    }
    else if (len == lpt.len)
    {
        lct.len = len;
        for (size_t i = 0; i < cipher_data.size(); i++)
        {
            Ciphertext ct;
            evaluator->multiply_plain(cipher_data[i], lpt.plain_data[i], ct);
            if (relin_keys != nullptr)
            {
                evaluator->relinearize_inplace(ct, *relin_keys);
            }

            lct.cipher_data.push_back(ct);
        }
    }
    else
    {
        char buf[100];
        sprintf(buf, "Length of BFVLongCiphertext(%ld) and BFVLongPlaintext(%ld) mismatch", len, lpt.len);
        throw bfv_lenth_error(buf);
    }
    return lct;
}

void BFVLongCiphertext::send(IOPack *io_pack, BFVLongCiphertext *lct)
{
    assert(lct->len > 0);
    io_pack->send_data(&(lct->len), sizeof(size_t));
    size_t size = lct->cipher_data.size();
    io_pack->send_data(&size, sizeof(size_t));
    for (size_t ct = 0; ct < size; ct++)
    {
        std::stringstream os;
        uint64_t ct_size;
        lct->cipher_data[ct].save(os);
        ct_size = os.tellp();
        string ct_ser = os.str();
        io_pack->send_data(&ct_size, sizeof(uint64_t));
        io_pack->send_data(ct_ser.c_str(), ct_ser.size());
    }
}

void BFVLongCiphertext::recv(IOPack *io_pack, BFVLongCiphertext *lct, SEALContext *context)
{
    io_pack->recv_data(&(lct->len), sizeof(size_t));
    size_t size;
    io_pack->recv_data(&size, sizeof(size_t));
    for (size_t ct = 0; ct < size; ct++)
    {
        Ciphertext cct;
        std::stringstream is;
        uint64_t ct_size;
        io_pack->recv_data(&ct_size, sizeof(uint64_t));
        char *c_enc_result = new char[ct_size];
        io_pack->recv_data(c_enc_result, ct_size);
        is.write(c_enc_result, ct_size);
        cct.unsafe_load(*context, is);
        lct->cipher_data.push_back(cct);
        delete[] c_enc_result;
    }
}