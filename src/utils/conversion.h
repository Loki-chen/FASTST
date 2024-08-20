#ifndef FAST_HE_BFV_CONVERSION_H__
#define FAST_HE_BFV_CONVERSION_H__

#include "FixedPoint/fixed-math.h"
#include "Utils/constants.h"
#include "he-bfv.h"
#include "mat-tools.h"
#include <ezpc_scilib/ezpc_utils.h>

class Conversion {
public:
    Conversion() {}

    ~Conversion() {}
    bfv_matrix he_to_ss_client(sci::NetIO *io, BFVKey *party);

    bfv_matrix he_to_ss_server(sci::NetIO *io, BFVParm *parm, const BFVLongCiphertext &in);

    void ss_to_he_client(BFVKey *party, NetIO *io, uint64_t *input, int length, int ell);

    BFVLongCiphertext ss_to_he_server(BFVParm *parm, NetIO *io, uint64_t *input, int length, int ell,
                                      bool is_add_share = true);

    void Prime_to_Ring(int party, const uint64_t *input, uint64_t *output, int length, int ell, int64_t plain_prime,
                       int s_in, int s_out, FPMath *fpmath);

    void Ring_to_Prime(const uint64_t *input, uint64_t *output, int length, int ell, int64_t plain_mod);

    void Ring_to_Prime(const uint64_t input, uint64_t output, int ell, int64_t plain_mod);

    void Prime_to_Ring(int party, int nthreads, const uint64_t *input, uint64_t *output, int length, int ell,
                       int64_t plain_prime, int s_in, int s_out, FPMath **fpmath);
};

#endif // FAST_HE_BFV_CONVERSION_H__