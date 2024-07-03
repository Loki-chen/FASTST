#include "fixed-ffh.h"

FixedFFN::FixedFFN(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
              Conversion *conv): FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv) {
    // load mat
}

BFVLongCiphertext FixedFFN::forward(const BFVLongCiphertext &input) const {
    return BFVLongCiphertext();
}