#ifndef FAST_FIXED_PROTOCOL_H
#define FAST_FIXED_PROTOCOL_H
#pragma once
#include "ezpc_scilib/ezpc_utils.h" // prg.h & io & arg
#include <model.h>
using sci::ALICE;
using sci::BOB;
using sci::IOPack;
using sci::OTPack;
using sci::PUBLIC;

#define DEFAULT_SCALE 12
#define DEFAULT_ELL 37

#define GELU_DEFAULT_SCALE 9
#define GELU_DEFAULT_ELL 22

class FixedProtocol
{
protected:
    BFVKey *party;
    BFVParm *parm;
    sci::NetIO *io;
    FPMath *fpmath;
    FPMath *fpmath_public;
    Conversion *conv;

public:
    FixedProtocol(BFVKey *party_, BFVParm *parm_, sci::NetIO *io_, FPMath *fpmath_,
                  FPMath *fpmath_public_, Conversion *conv_) : party(party_),
                                                               parm(parm_),
                                                               io(io_),
                                                               fpmath(fpmath_),
                                                               fpmath_public(fpmath_public_),
                                                               conv(conv_)
    {
        assert(party->party == fpmath->fix->party);
    }
    ~FixedProtocol() {}
};
#endif