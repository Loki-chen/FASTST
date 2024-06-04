#ifndef FAST_FIXED_TOOLS_H__
#define FAST_FIXED_TOOLS_H__
#pragma once
#include <iostream>
#include <random>
#include <vector>

class FixedArray
{
public:
    int size = 0;
    uint64_t *data = nullptr;
    bool signed_;
    int bitlength;
    int scale;

    FixedArray() {}

    FixedArray(int size_, bool signed__, int bitlength_, int scale_)
    {
        assert(size_ > 0);
        assert(bitlength_ <= 64 && bitlength_ > 0);
        this->size = size;
        this->signed_ = signed__;
        this->bitlength = bitlength_;
        this->scale = scale_;
        data = new uint64_t[size];
    }

    ~FixArray() { delete[] data; }
    uint64_t bitlen_mask() const { return ((bitlength == 64) ? -1 : (1ULL << (this->bitlength)) - 1); }
};

int64_t getSignedVal(uint64_t x, uint64_t mod);
uint64_t getRingElt(int64_t x, uint64_t mod);
uint64_t FixedAdd(uint64_t x, uint64_t y, uint64_t mod);
uint64_t FixedSub(uint64_t x, uint64_t y, uint64_t mod);
uint64_t FixedMult(uint64_t x, uint64_t y, uint64_t mod);
uint64_t FixedDiv(uint64_tx x, uint64_t y, uint64_t mod);

#endif