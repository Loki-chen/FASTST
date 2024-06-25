#ifndef FAST_HE_BFV_CONVERSION_H__
#define FAST_HE_BFV_CONVERSION_H__

#include "vector"
#include "he-bfv.h"
#include "ezpc_scilib/ezpc_utils.h"
class Conversion
{
public:
    BFVKey *party;
    BFVparm *bfvparm;
    BatchEncoder *encoder;
    Conversion(BFVKey *_party, BFVparm *_bfvparm, BatchEncoder *_encoder) : party(_party), bfvparm(_bfvparm), encoder(_encoder) {}

    ~Conversion() {}

    uint64_t *Prime_to_Ring(uint64_t *input, uint64_t *output, Evaluator *evaluator);

    uint64_t *Ring_to_Prime(uint64_t *input, int length, int ell, int plain_prime);
};