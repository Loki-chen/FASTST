#ifndef FAST_PROTOCOL_H
#define FAST_PROTOCOL_H
#pragma once
#include <model.h>
#include "ezpc_scilib/ezpc_utils.h" // prg.h & io & arg
class Protocol
{
protected:
    CKKSKey *party;
    CKKSEncoder *encoder;
    Evaluator *evaluator;
    sci::NetIO *io;

public:
    Protocol(CKKSKey *party_, CKKSEncoder *encoder_, Evaluator *evaluator_,
             sci::NetIO *io_) : party(party_),
                                encoder(encoder_),
                                evaluator(evaluator_),
                                io(io_) {}
    ~Protocol() {}
};
#endif