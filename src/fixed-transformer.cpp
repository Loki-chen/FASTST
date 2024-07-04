#include "fixed-transformer.h"

FixedEncoder::FixedEncoder(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath,
                           FPMath *fpmath_public, Conversion *conv) {
    multi_head_attn = new Fixed_Multi_Head_Attention(layer, party, parm, io, fpmath, fpmath_public, conv);
    ln1 = new FixedLayerNorm(layer, party, parm, io, fpmath, fpmath_public, conv, true);
}

FixedEncoder::~FixedEncoder() {
    delete multi_head_attn;
    delete ln1;
}

bfv_matrix FixedEncoder::forward(const bfv_matrix &input) {
    BFVLongCiphertext output1 = multi_head_attn->forward(input);
    BFVLongCiphertext output2 = ln1->forward(output1, input);
    return bfv_matrix();
}