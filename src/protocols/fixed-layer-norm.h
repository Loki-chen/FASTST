#ifndef FAST_FIXED_LAYER_NROM1_H__
#define FAST_FIXED_LAYER_NROM1_H__
#include "fixed-protocol.h"

class FixedLayerNorm : public FixedProtocol {
    bool after_attn;
    bfv_matrix gamma, beta;

public:
    FixedLayerNorm(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                   Conversion *conv, bool after_attn);
    ~FixedLayerNorm() {}
    // Alice possess: attn_secret_b X_a, Bob possess X_b
    BFVLongCiphertext forward(const BFVLongCiphertext &attn, const bfv_matrix &input) const;
};
#endif
