#ifndef FAST_FIXED_PROTOCOL_H
#define FAST_FIXED_PROTOCOL_H

#include "ezpc_scilib/ezpc_utils.h" // prg.h & io & arg
#include <model.h>

#define DEFAULT_SCALE 12
#define DEFAULT_ELL 37

#define GELU_DEFAULT_SCALE 9
#define GELU_DEFAULT_ELL 22

class FixedProtocol {
protected:
    BFVKey *party;
    BFVParm *parm;
    FPMath *fpmath;
    FPMath *fpmath_public;

public:
    FixedProtocol(BFVKey *party_, BFVParm *parm_, FPMath *fpmath_,
                  FPMath *fpmath_public_)
        : party(party_), parm(parm_), fpmath(fpmath_),
          fpmath_public(fpmath_public_) {
        assert(party->party == fpmath->fix->party);
    }
    ~FixedProtocol() {}
};
#endif