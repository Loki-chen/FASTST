#include "fixed_lib.h"

int64_t getSignedVal(uint64_t x, uint64_t mod)
{
    assert(x < mod);
    bool xPos;
    if (mod & 1)
        xPos = (x <= (mod / 2));
    else
        xPos = (x < (mod / 2));
    int64_t sx = x;
    if (!xPos)
        sx = x - mod;
    return sx;
}

uint64_t getRingElt(int64_t x, uint64_t mod)
{
    if (x > 0)
        return x % mod;
    else
    {
        int64_t y = -x;
        int64_t temp = mod - y;
        int64_t temp1 = temp % ((int64_t)mod);
        uint64_t ans = (temp1 + mod) % mod;
        return ans;
    }
}

uint64_t FixedAdd(uint64_t x, uint64_t y, uint64_t mod)
{
    assert((x < mod) && (y < mod));
    return (x + y) % mod;
}

uint64_t PublicSub(uint64_t x, uint64_t y, uint64_t mod)
{
    assert((x < mod) && (y < mod));
    uint64_t ans;
    if (x >= y)
        ans = (x - y) % mod;
    else
        ans = ((x + mod) - y) % mod;
    return ans;
}

uint64_t FixedMult(uint64_t x, uint64_t y, uint64_t mod)
{
    assert((x < mod) && (y < mod));
#ifdef __SIZEOF_INT128__
    __int128 ix = x;
    __int128 iy = y;
    __int128 iz = ix * iy;

    return iz % mod;
#else
    uint64_t res = 0;
    a %= mod;
    while (b)
    {
        if (b & 1)
            res = (res + a) % mod;
        a = (2 * a) % mod;
        b >>= 1;
    }
    return res;
#endif
}

uint64_t FixedDiv(uint64_t x, uint64_t y, uint64_t mod)
{
    int64_t sign_x = getSignedVal(x, mod);
    int64_t sign_y = getSignedVal(y, mod);
    int64_t q, r;
}