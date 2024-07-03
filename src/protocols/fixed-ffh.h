
#ifndef FAST_FIXED_FFN_H__
#define FAST_FIXED_FFN_H__
#include "fixed-protocol.h"
#include "utils/he-bfv.h"

class FixedFFN : public FixedProtocol {
    bfv_matrix W1, W2, b1, b2;
public:
    FixedFFN(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
              Conversion *conv);
    ~FixedFFN() {}
    BFVLongCiphertext forward(const BFVLongCiphertext &input) const;
};

#endif // FAST_FIXED_FFN_H__