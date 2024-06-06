#ifndef FIXED_POINT_H__
#define FIXED_POINT_H__

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

class FixArry
{
public:
    int party = PUBLIC;
    int size = 0;             // size of array
    uint64_t *data = nullptr; // data (ell-bit integers)
    bool signed_;             // signed? (1: signed; 0: unsigned)
    int ell;                  // bitlength
    int s;                    // scale

    FixArray(){};

    FixArray(int party_, int sz, bool signed__, int ell_, int s_ = 0)
    {
        assert(party_ == PUBLIC || party_ == ALICE || party_ == BOB);
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

    template <class T>
    std::vector<T> get_native_type();

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

std::ostream &operator<<(std::ostream &os, FixArray &other);

class FixOp
{

public:
    int party;
};