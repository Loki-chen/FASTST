#ifndef FAST_FIXED_TRANSFORMER_H__
#define FAST_FIXED_TRANSFORMER_H__
#include "Utils/net_io_channel.h"
#include "protocols.h"
#include "utils/conversion.h"
#include "utils/he-bfv.h"

class FixedEncoder {
    int layer;
    sci::NetIO *io;
    BFVKey *party;
    Conversion *conv;
    Fixed_Multi_Head_Attention *multi_head_attn;
    FixedLayerNorm *ln1;
    FixedFFN *ffn;
    FixedLayerNorm *ln2;

public:
    FixedEncoder(int _layer, BFVKey *_party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                 Conversion *_conv);
    ~FixedEncoder();
    bfv_matrix forward(const bfv_matrix &input);
};

class FixedTransformer {
    FixedEncoder **layer;

public:
    FixedTransformer(BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                     Conversion *conv);
    ~FixedTransformer();
    bfv_matrix forward(const bfv_matrix &input);
};

#endif