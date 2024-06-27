#ifndef FAST_HE_BFV_CONVERSION_H__
#define FAST_HE_BFV_CONVERSION_H__

#include "vector"
#include "he-bfv.h"
#include "ezpc_scilib/ezpc_utils.h"
class Conversion
{
public:
    Conversion() {}

    ~Conversion() {}

    uint64_t *Prime_to_Ring(int party, uint64_t *input, int length, int ell, u_int64_t plain_prime, int s_in, int s_out, FPMath *fpmath);

    uint64_t *Ring_to_Prime(uint64_t *input, int length, int ell, int64_t plain_prime);
};

#endif // FAST_HE_BFV_CONVERSION_H__