#ifndef FIXED_POINT_H__
#define FIXED_POINT_H__
#pragma once
#include "OT/emp-ot.h"
#include "Millionaire/millionaire_with_equality.h"
#include "Millionaire/equality.h"
#include "BuildingBlocks/aux-protocols.h"
#include "BuildingBlocks/value-extension.h"
#include "BuildingBlocks/truncation.h"
#include "BuildingBlocks/linear-ot.h"
#include "bool-data.h"
#include "Math/math-functions.h"

#endif
#define print_fix(vec)                               \
    {                                                \
        auto tmp_pub = fix->output(PUBLIC, vec);     \
        cout << #vec << "_pub: " << tmp_pub << endl; \
    }

// A container to hold an array of fixed-point values
// If party is set as PUBLIC for a FixArray instance, then the underlying array is known publicly and we maintain the invariant that both parties will hold identical data in that instance.
// Else, the underlying array is secret-shared and the class instance will hold the party's share of the secret array. In this case, the party data member denotes which party this share belongs to.
// signed_ denotes the signedness, ell is the bitlength, and s is the scale of the underlying fixed-point array
// If s is set to 0, the FixArray will behave like an IntegerArray

class FixArray
{
public:
    int party = sci::PUBLIC;
    int size = 0;             // size of array
    uint64_t *data = nullptr; // data (ell-bit integers)
    bool signed_;             // signed? (1: signed; 0: unsigned)
    int ell;                  // bitlength
    int s;                    // scale

    FixArray(){};

    FixArray(int party_, int sz, bool signed__, int ell_, int s_ = 0)
    {
        assert(party_ == sci::PUBLIC || party_ == sci::ALICE || party_ == sci::BOB);
        assert(sz > 0);
        assert(ell_ <= 64 && ell_ > 0);
        this->party = party_;
        this->size = sz;
        this->signed_ = signed__;
        this->ell = ell_;
        this->s = s_;
        data = new uint64_t[sz];
    }

    // copy constructor
    FixArray(const FixArray &other)
    {
        this->party = other.party;
        this->size = other.size;
        this->signed_ = other.signed_;
        this->ell = other.ell;
        this->s = other.s;
        this->data = new uint64_t[size];
        memcpy(this->data, other.data, size * sizeof(uint64_t));
    }

    // move constructor
    FixArray(FixArray &&other) noexcept
    {
        this->party = other.party;
        this->size = other.size;
        this->signed_ = other.signed_;
        this->ell = other.ell;
        this->s = other.s;
        this->data = other.data;
        other.data = nullptr;
    }

    ~FixArray() { delete[] data; }

    // copy assignment
    FixArray &operator=(const FixArray &other)
    {
        if (this == &other)
            return *this;

        delete[] this->data;
        this->party = other.party;
        this->size = other.size;
        this->signed_ = other.signed_;
        this->ell = other.ell;
        this->s = other.s;
        this->data = new uint64_t[size];
        memcpy(this->data, other.data, size * sizeof(uint64_t));
        return *this;
    }

    // move assignment
    FixArray &operator=(FixArray &&other) noexcept
    {
        if (this == &other)
            return *this;

        delete[] this->data;
        this->party = other.party;
        this->size = other.size;
        this->signed_ = other.signed_;
        this->ell = other.ell;
        this->s = other.s;
        this->data = other.data;
        other.data = nullptr;
        return *this;
    }

    uint64_t ell_mask() const { return ((ell == 64) ? -1 : (1ULL << (this->ell)) - 1); }
};

class FixOp
{

public:
    int party;
    sci::IOPack *iopack;
    sci::OTPack *otpack;
    Equality *eq;
    MillionaireWithEquality *mill_eq;
    AuxProtocols *aux;
    XTProtocol *xt;
    Truncation *trunc;
    LinearOT *mult;
    BoolOp *bool_op;
    FixOp *fix;

    FixOp(int party, sci::IOPack *iopack, sci::OTPack *otpack)
    {
        this->party = party;
        this->iopack = iopack;
        this->otpack = otpack;
        this->aux = new AuxProtocols(party, iopack, otpack);
        this->eq = new Equality(party, iopack, otpack);
        this->mill_eq = new MillionaireWithEquality(party, iopack, otpack);
        this->xt = new XTProtocol(party, iopack, otpack);
        this->trunc = new Truncation(party, iopack, otpack);
        this->mult = new LinearOT(party, iopack, otpack);
        this->bool_op = new BoolOp(party, iopack, otpack);
        this->fix = this;
    }

    ~FixOp()
    {
        delete aux;
        delete eq;
        delete mill_eq;
        delete xt;
        delete trunc;
        delete mult;
        delete bool_op;
    }

    // input functions: return a FixArray that stores data_
    // party_ denotes which party provides the input data_ and the data_ provided by the other party is ignored. If party_ is PUBLIC, then the data_ provided by both parties must be identical.
    // sz is the size of the returned FixArray and the uint64_t array pointed by data_
    // signed__, ell_, and s_ are the signedness, bitlength and scale of the input, respectively
    FixArray input(int party_, int sz, uint64_t *data_, bool signed__, int ell_, int s_ = 0);
    // same as the above function, except that it replicates data_ in all sz positions of the returned FixArray
    FixArray input(int party_, int sz, uint64_t data_, bool signed__, int ell_, int s_ = 0);

    // output function: returns the secret array underlying x in the form of a PUBLIC FixArray
    // party_ denotes which party will receive the output. If party_ is PUBLIC, both parties receive the output.
    FixArray output(int party_, const FixArray &x);
};