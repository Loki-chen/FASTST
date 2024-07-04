
#ifndef FAST_FIXED_ATTENTION_H__
#define FAST_FIXED_ATTENTION_H__
#include "fixed-protocol.h"
#include "utils/he-bfv.h"

class Fixed_Multi_Head_Attention;

class Fixed_Attention : public FixedProtocol {
    int head;
    bfv_matrix WQ, WK, WV, bQ, bK, bV;

public:
    friend Fixed_Multi_Head_Attention;
    Fixed_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                    Conversion *conv, int head_);
    ~Fixed_Attention() {}
    bfv_matrix forward(const bfv_matrix &input) const;
};

class Fixed_Multi_Head_Attention : public FixedProtocol {
    int layer;
    bfv_matrix W, b;
    Fixed_Attention **attns;

public:
    Fixed_Multi_Head_Attention(int layer, BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath,
                               FPMath *fpmath_public, Conversion *conv);
    ~Fixed_Multi_Head_Attention();
    BFVLongCiphertext forward(const bfv_matrix &input) const;
};

#endif // FAST_FIXED_ATTENTION_H__