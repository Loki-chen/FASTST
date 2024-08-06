#include "fixed-point-field.h"

template <class T>
vector<T> FixFieldArray::get_native_field_type() const
{
    assert(this->party == PUBLIC);

    vector<T> ret(this->size);
    double den = pow(2.0, this->scale);
    for (int i = 0; i < this->size; i++)
    {
        int64_t data_ = this->data[i];
        ret[i] = T(data_ / den);
    }
    return ret;
}

template vector<int64_t> FixFieldArray::get_native_field_type() const;
template vector<float> FixFieldArray::get_native_field_type() const;
template vector<double> FixFieldArray::get_native_field_type() const;

FixFieldArray FixFieldOp::input(int party_, int size, const int64_t *data_, int field_, int scale_)
{
    FixFieldArray ret((party_ == PUBLIC ? party_ : this->party), size, field_, scale_);
    // uint64_t ell_mask_ = ret.ell_mask();
    if ((this->party == party_) || (party_ == PUBLIC))
    {
        memcpy(ret.data, data_, size * sizeof(int64_t));
        // for (int i = 0; i < size; i++)
        // {
        //     ret.data[i] &= ell_mask_;
        // }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ret.data[i] = 0;
        }
    }
    return ret;
}

FixFieldArray FixFieldOp::input(int party_, int size, int64_t data_, int field_, int scale_)
{
    FixFieldArray ret((party_ == PUBLIC ? party_ : this->party), size, field_, scale_);
    // uint64_t ell_mask_ = ret.ell_mask();
    if ((this->party == party_) || (party_ == PUBLIC))
    {
        for (int i = 0; i < size; i++)
        {
            ret.data[i] = data_;
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ret.data[i] = 0;
        }
    }
    return ret;
}

FixFieldArray FixFieldOp::output(int party_, const FixFieldArray &x)
{
    if (x.party == PUBLIC)
    {
        return x;
    }
    int size = x.size;
    int ret_party = (party_ == PUBLIC || party_ == x.party ? PUBLIC : x.party);
    FixFieldArray ret(ret_party, size, x.field, x.scale);
#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 1 && party_ != BOB)
        {
            if (party == ALICE)
            {
                iopack->io_rev->recv_data(ret.data, size * sizeof(int64_t));
            }
            else
            { // party == BOB
                iopack->io_rev->send_data(x.data, size * sizeof(int64_t));
            }
        }
        else if (omp_get_thread_num() == 0 && party_ != ALICE)
        {
            if (party == ALICE)
            {
                iopack->io->send_data(x.data, size * sizeof(int64_t));
            }
            else
            { // party == BOB
                iopack->io->recv_data(ret.data, size * sizeof(int64_t));
            }
        }
    }

    for (int i = 0; i < size; i++)
    {
        ret.data[i] = (ret.data[i] + x.data[i]);
    }
    return ret;
}

FixFieldArray FixFieldOp::add(const FixFieldArray &x, const FixFieldArray &y)
{
    assert(x.size == y.size);

    assert(x.field == y.field);
    assert(x.scale == y.scale);

    bool x_cond, y_cond;
    int party_;
    if (x.party == PUBLIC && y.party == PUBLIC)
    {
        x_cond = false;
        y_cond = false;
        party_ = PUBLIC;
    }
    else
    {
        x_cond = (x.party == PUBLIC) && (this->party == BOB);
        y_cond = (y.party == PUBLIC) && (this->party == BOB);
        party_ = this->party;
    }
    FixFieldArray ret(party_, x.size, x.field, x.scale);
    for (int i = 0; i < x.size; i++)
    {
        ret.data[i] = ((x_cond ? 0 : x.data[i]) + (y_cond ? 0 : y.data[i]));
    }
    return ret;
}

FixFieldArray FixFieldOp::add(const FixFieldArray &x, int64_t y)
{
    FixFieldArray y_fix = this->input(PUBLIC, x.size, y, x.field, x.scale);
    return this->add(x, y_fix);
}

FixFieldArray FixFieldOp::mul(const FixFieldArray &x, const FixFieldArray &y, int field)
{
    assert(x.party == ALICE && y.party == ALICE || x.party == BOB && y.party == BOB ||
           x.party == PUBLIC && y.party == PUBLIC);
    assert(x.size == y.size);
    assert(field >= x.field && field >= y.field && field <= x.field + y.field);
    assert(field < (3 * field));
    FixFieldArray ret(x.party, x.size, x.field, x.scale + y.scale);

    FixFieldArray x_ext(x.party, x.size, x.field, x.scale);
    FixFieldArray y_ext(y.party, y.size, y.field, y.scale);

    for (size_t i = 0; i < x.size; i++)
    {
        ret.data[i] = (x.data[i] * y.data[i]);
    }
    return ret;
}