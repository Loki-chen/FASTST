#include "fixed-transformer.h"
#include "utils/he-bfv.h"

FixedEncoder::FixedEncoder(int _layer, BFVKey *_party, BFVParm *parm, sci::NetIO *_io, FPMath *fpmath,
                           FPMath *fpmath_public, Conversion *_conv)
    : layer(_layer), party(_party), io(_io), conv(_conv) {
    multi_head_attn = new Fixed_Multi_Head_Attention(layer, party, parm, io, fpmath, fpmath_public, conv);
    ln1 = new FixedLayerNorm(layer, party, parm, io, fpmath, fpmath_public, conv, true);
    ffn = new FixedFFN(layer, party, parm, io, fpmath, fpmath_public, conv);
    ln2 = new FixedLayerNorm(layer, party, parm, io, fpmath, fpmath_public, conv, false);
}

FixedEncoder::~FixedEncoder() {
    delete multi_head_attn;
    delete ln1;
    delete ffn;
    delete ln2;
}

bfv_matrix FixedEncoder::forward(const bfv_matrix &input) {
    size_t total_comm = io->counter;
#ifdef LOG
    INIT_TIMER
    START_TIMER
#endif
    BFVLongCiphertext output1 = multi_head_attn->forward(input);
    BFVLongCiphertext output2 = ln1->forward(output1, input);
    BFVLongCiphertext output3 = ffn->forward(output2);
    BFVLongCiphertext output4 = ln2->forward(output3, input);
    bfv_matrix ret;
    if (party->party == sci::ALICE) {
        ret = conv->he_to_ss_client(io, party);
    } else {
        ret = conv->he_to_ss_server(io, party->parm, output4);
    }
#ifdef LOG
    char *buf = new char[9];
    sprintf(buf, "Layer-%-2d", layer);
    STOP_TIMER(buf)
    total_comm += io->counter;
    printf("%s Send data %ld Bytes. \n", buf, total_comm);
    delete[] buf;
#endif
    return ret;
}

FixedTransformer::FixedTransformer(BFVKey *party, BFVParm *parm, sci::NetIO *io, FPMath *fpmath, FPMath *fpmath_public,
                                   Conversion *conv) {
    layer = new FixedEncoder *[n_layer];
    for (int i = 0; i < n_layer; i++) {
        layer[i] = new FixedEncoder(i, party, parm, io, fpmath, fpmath_public, conv);
    }
}

FixedTransformer::~FixedTransformer() {
    for (int i = 0; i < n_heads; i++) {
        delete layer[i];
    }
    delete[] layer;
}

bfv_matrix FixedTransformer::forward(const bfv_matrix &input) {
    bfv_matrix output = layer[0]->forward(input);
    for (int i = 1; i < n_heads; i++) {
        output = layer[i]->forward(output);
    }
    return output;
}