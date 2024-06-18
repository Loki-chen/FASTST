#ifndef FAST_FIXED_PROTOCOL_H
#define FAST_FIXED_PROTOCOL_H
#pragma once
#include <model.h>
#include "ezpc_scilib/ezpc_utils.h" // prg.h & io & arg
using sci::ALICE;
using sci::BOB;
using sci::IOPack;
using sci::OTPack;
using sci::PUBLIC;

class FixedProtocol
{
protected:
    BFVKey *party;
    BFVParm *parm;
    FixOp *fixop;
    FixOp *fix_public;

public:
    FixedProtocol(BFVKey *party_, BFVParm *parm_,
                  FixOp *fixop_, FixOp *fix_public_) : party(party_), parm(parm_),
                                                       fixop(fixop_), fix_public(fix_public_)
    {
        assert(party->party == fixop->party);
    }
    ~FixedProtocol() {}
};
#endif