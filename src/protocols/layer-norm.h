
#ifndef FAST_LAYER_NROM1_H__
#define FAST_LAYER_NROM1_H__
#include "protocol.h"
#pragma once
class LayerNorm : public Protocol
{
public:
    LayerNorm(CKKSKey *party, CKKSEncoder *encoder, Evaluator *evaluator,
              IOPack *io_pack) : Protocol(party, encoder, evaluator, io_pack) {}
    ~LayerNorm() {}
    LongCiphertext forward(const LongCiphertext &attn, const matrix &input) const;
};
#endif
