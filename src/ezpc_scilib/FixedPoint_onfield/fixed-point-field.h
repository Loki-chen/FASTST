#ifndef FIXED_POINT_FIELD_H__
#define FIXED_POINT_FIELD_H__

#include "cmath"
#include "OT/emp-ot.h"
#include "omp.h"

using namespace std;
using namespace sci;

// A container to hold an array of fixed-point values on the field P (prime)

// const int64_t bound = (1ULL << 64) - 1; // next_prime(2^64)

class FixFieldArray
{
public:
    int party = PUBLIC;
    int size = 0;            // size of array
    int64_t *data = nullptr; // data (field-bit integers)
    // bool signed_;            // signed? (1: Positive ; 0: negative)
    int64_t field; // field
    int scale;     // scale

    FixFieldArray() {};

    FixFieldArray(int party_, int size_, int field_, int scale_ = 0)
    {
        assert(party_ == PUBLIC || party_ == ALICE || party_ == BOB);
        assert(size_ > 0);
        assert(field_ <= (3 * field_) && field_ > 0);
        this->party = party_;
        this->size = size_;
        this->field = field_;
        this->scale = scale_;
        data = new int64_t[size_];
    }

    // copy constructor
    FixFieldArray(const FixFieldArray &other)
    {
        this->party = other.party;
        this->size = other.size;
        this->field = other.field;
        this->scale = other.scale;
        this->data = new int64_t[size];
        memcpy(this->data, other.data, size * sizeof(int64_t));
    }

    // move constructor
    FixFieldArray(FixFieldArray &&other) noexcept
    {
        this->party = other.party;
        this->size = other.size;
        this->field = other.field;
        this->scale = other.scale;
        this->data = other.data;
        other.data = nullptr;
    }

    ~FixFieldArray() { delete[] data; }

    template <class T>
    vector<T> get_native_field_type() const;

    // copy assignment
    FixFieldArray &operator=(const FixFieldArray &other)
    {
        if (this == &other)
            return *this;
        delete[] this->data;
        this->party = other.party;
        this->size = other.size;
        this->field = other.field;
        this->scale = other.scale;
        this->data = new int64_t[size];
        memcpy(this->data, other.data, size * sizeof(int64_t));
        return *this;
    }

    // move assignment
    FixFieldArray &operator=(FixFieldArray &&other) noexcept
    {
        if (this == &other)
            return *this;
        delete[] this->data;
        this->party = other.party;
        this->size = other.size;
        this->field = other.field;
        this->scale = other.scale;
        this->data = other.data;
        other.data = nullptr;
        return *this;
    }
};

class FixFieldOp
{

public:
    int party;
    IOPack *iopack;
    OTPack *otpack;
    FixFieldOp *fixfield;

    FixFieldOp(int party, IOPack *iopack, OTPack *otpack)
    {
        this->party = party;
        this->iopack = iopack;
        this->otpack = otpack;
        this->fixfield = this;
    }

    ~FixFieldOp() {}

    // input functions: return a FixFieldArray that stores data_
    // party_ denotes which party provides the input data_ and the data_ provided by the other party is ignored. If party_ is PUBLIC, then the data_ provided by both parties must be identical.
    // sz is the size of the returned FixFieldArray and the uint64_t array pointed by data_
    // signed__, field_, and scale_ are the signedness, bitlength and scale of the input, respectively
    FixFieldArray input(int party_, int sz, const int64_t *data_, int field_, int scale_ = 0);
    // // same as the above function, except that it replicates data_ in all sz positions of the returned FixFieldArray
    FixFieldArray input(int party_, int sz, int64_t data_, int field_, int scale_ = 0);
    // output function: returns the secret array underlying x in the form of a PUBLIC FixFieldArray
    // party_ denotes which party will receive the output. If party_ is PUBLIC, both parties receive the output.
    FixFieldArray output(int party_, const FixFieldArray &x);

    // Add Operations: return x[i] + y[i]
    // The output has same signedness, bitlength and scale as x, y
    //// Both x and y can be PUBLIC or secret-shared
    //// x and y must have equal size
    //// x, y must have same signedness, bitlength and scale
    FixFieldArray add(const FixFieldArray &x, const FixFieldArray &y);
    //// x can be PUBLIC or secret-shared
    //// y[i] = y (with same signedness, bitlength and scale as x)
    FixFieldArray add(const FixFieldArray &x, int64_t y);

    // // Sub Operations: return x[i] - y[i]
    // // The output has same signedness, bitlength and scale as x, y
    // //// Both x and y can be PUBLIC or secret-shared
    // //// x and y must have equal size
    // //// x, y must have same signedness, bitlength and scale
    // FixFieldArray sub(const FixFieldArray &x, const FixFieldArray &y);
    // //// x can be PUBLIC or secret-shared
    // //// y[i] = y (with same signedness, bitlength and scale as x)
    // FixFieldArray sub(const FixFieldArray &x, uint64_t y);
    // //// y can be PUBLIC or secret-shared
    // //// x[i] = x (with same signedness, bitlength and scale as y)
    // FixFieldArray sub(uint64_t x, const FixFieldArray &y);

    // // Extension: returns a FixFieldArray that holds the same fixed-point values as x, except in larger bitlength field
    // // The signedness and scale of the output are same as x
    // // x can be PUBLIC or secret-shared
    // // field should be greater than or equal to bitlength of x
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper when x is secret-shared.
    // FixFieldArray extend(const FixFieldArray &x, int field, uint8_t *msb_x = nullptr);

    // Multiplication Operation: return x[i] * y[i] (in field bits)
    // field specifies the output bitlength
    // Signedness of the output is the same as x and scale is equal to sum of scales of x and y
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper when x is secret-shared.
    // //// At least one of x and y must be a secret-shared FixFieldArray
    // //// x and y must have equal size
    // //// Either signedness of x and y is same, or x is signed (x.signed_ = 1)
    // //// field >= bitlengths of x and y and field <= sum of bitlengths of x and y (x.field + y.field)
    // //// msb_y is similar to msb_x but for y
    // FixFieldArray mul(const FixFieldArray &x, const FixFieldArray &y, int field,
    //                   uint8_t *msb_x = nullptr, uint8_t *msb_y = nullptr);
    // //// x can be PUBLIC or secret-shared
    // //// y[i] = y (with same signedness as x; bitlength is field and scale is 0)
    // //// field >= bitlength of x

    FixFieldArray mul(const FixFieldArray &x, const FixFieldArray &y, int field);

    // // Left Shift: returns x[i] << s[i] (in field bits)
    // // Output bitlength is field, output signedness and scale are same as that of x
    // // bound is the (closed) upper bound on integer values in s
    // // Both x and s must be secret shared FixFieldArray and of equal size
    // // field <= bitlength of x (x.field) + bound and field is >= both x.field and bound
    // // s must be an unsigned FixFieldArray with scale 0 and bitlength >= ceil(log2(bound))
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper
    // FixFieldArray left_shift(const FixFieldArray &x, const FixFieldArray &s, int field, int bound,
    //                          uint8_t *msb_x = nullptr);

    // // Right Shift: returns x[i] >> s[i]
    // // Output bitlength, signedness and scale are same as that of x
    // // bound is the (closed) upper bound on integer values in s
    // // Both x and s must be secret shared FixFieldArray and of equal size
    // // bound <= bitlength of x (x.field) and bound + x.field < 64
    // // s must be an unsigned FixFieldArray with scale 0 and bitlength >= ceil(log2(bound))
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper
    // FixFieldArray right_shift(const FixFieldArray &x, const FixFieldArray &s, int bound,
    //                           uint8_t *msb_x = nullptr);

    // // Right Shift: returns x[i] >> s
    // // Output scale is x.s - s; Output bitlength and signedness are same as that of x
    // // x must be secret shared FixFieldArray
    // // s <= bitlength of x (x.field) and s >= 0
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper
    // FixFieldArray right_shift(const FixFieldArray &x, int s, uint8_t *msb_x = nullptr);
    // // Right Shift: returns x[i] >> s
    // // Output scale is x.s - s; Output bitlength and signedness are same as that of x
    // // x must be secret shared FixFieldArray
    // // s <= bitlength of x (x.field) and s >= 0
    // // msb_x is an optional parameter that points to an array holding boolean shares of most significant bit (MSB) of x[i]'s. If msb_x provided, this operation is cheaper
    // FixFieldArray location_right_shift(const FixFieldArray &x, int s, uint8_t *msb_x = nullptr);

    // // Scale-Up: returns (x[i] << (s - x.s)) mod p^{field}
    // // Output bitlength and scale are field and s, and output signedness is same as that of x
    // // x can be PUBLIC or secret-shared
    // // s >= x.s and field <= x.field + (s - x.s)
    // FixFieldArray scale_up(const FixFieldArray &x, int field, int s);

    // // (Modulo) Reduce: returns x[i] mod p^{field}
    // // Output bitwidth is field, and output scale and signedness are same as that of x
    // // x can be PUBLIC or secret-shared
    // // field <= x.field and field > 0
    // FixFieldArray reduce(const FixFieldArray &x, int field);

    // // Least Significant Bit: returns x[i] mod 2 in the form of BoolArray
    // // x can be PUBLIC or secret-shared
    // BoolArray LSB(const FixFieldArray &x);

    // // Truncate and Reduce: returns x[i] >> s mod 2^{x.field - s}
    // // Output bitlength and scale are x.field-s and x.s-s; Output signedness is same as that of x
    // // x must be public FixFieldArray
    // // s < bitlength of x (x.field) and s >= 0
    // FixFieldArray location_truncation(const FixFieldArray &x, int scale);

    // BoolArray wrap(const FixFieldArray &x);

    // void send_fix_field_array(const FixFieldArray &fix_array);

    // void recv_fix_field_array(FixFieldArray &fix_array);
};
#endif