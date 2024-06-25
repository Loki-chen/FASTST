#ifndef FAST_FIXED_LAYER_NROM1_H__
#define FAST_FIXED_LAYER_NROM1_H__
#include "fixed-protocol.h"

using namespace sci;
using namespace std;
class FixedLayerNorm : public FixedProtocol
{
public:
    FixedLayerNorm(BFVKey *party, BFVParm *parm,
                   FPMath *fpmath, FPMath *fpmath_public) : FixedProtocol(party, parm, fpmath, fpmath_public) {}
    ~FixedLayerNorm() {}
    // Alice possess: attn_secret_b X_a, Bob possess X_b
    BFVLongCiphertext forward(const BFVLongCiphertext &attn, const bfv_matrix &input) const;
};
#endif
