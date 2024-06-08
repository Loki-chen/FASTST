#include "fixed-point.h"

using namespace std;

FixArray FixOp::input(int party_, int sz, uint64_t *data_, bool signed__, int ell_, int s_)
{
    FixArray ret((party_ == sci::PUBLIC ? party_ : this->party), sz, signed__, ell_, s_);
    uint64_t ell_mask_ = ret.ell_mask();
    if ((this->party == party_) || (party_ == sci::PUBLIC))
    {
        memcpy(ret.data, data_, sz * sizeof(uint64_t));
        for (int i = 0; i < sz; i++)
        {
            ret.data[i] &= ell_mask_;
        }
    }
    else
    {
        for (int i = 0; i < sz; i++)
        {
            ret.data[i] = 0;
        }
    }
    return ret;
}

FixArray FixOp::input(int party_, int sz, uint64_t data_, bool signed__, int ell_, int s_)
{
    FixArray ret((party_ == sci::PUBLIC ? party_ : this->party), sz, signed__, ell_, s_);
    uint64_t ell_mask_ = ret.ell_mask();
    if ((this->party == party_) || (party_ == sci::PUBLIC))
    {
        for (int i = 0; i < sz; i++)
        {
            ret.data[i] = data_ & ell_mask_;
        }
    }
    else
    {
        for (int i = 0; i < sz; i++)
        {
            ret.data[i] = 0;
        }
    }
    return ret;
}

FixArray FixOp::output(int party_, const FixArray &x)
{
    if (x.party == sci::PUBLIC)
    {
        return x;
    }
    int sz = x.size;
    int ret_party = (party_ == sci::PUBLIC || party_ == x.party ? sci::PUBLIC : x.party);
    FixArray ret(ret_party, sz, x.signed_, x.ell, x.s);
#pragma omp parallel num_threads(2)
    {
        if (omp_get_thread_num() == 1 && party_ != sci::BOB)
        {
            if (party == sci::ALICE)
            {
                iopack->io_rev->recv_data(ret.data, sz * sizeof(uint64_t));
            }
            else
            { // party == sci::BOB
                iopack->io_rev->send_data(x.data, sz * sizeof(uint64_t));
            }
        }
        else if (omp_get_thread_num() == 0 && party_ != sci::ALICE)
        {
            if (party == sci::ALICE)
            {
                iopack->io->send_data(x.data, sz * sizeof(uint64_t));
            }
            else
            { // party == sci::BOB
                iopack->io->recv_data(ret.data, sz * sizeof(uint64_t));
            }
        }
    }
    uint64_t ell_mask_ = x.ell_mask();
    for (int i = 0; i < sz; i++)
    {
        ret.data[i] = (ret.data[i] + x.data[i]) & ell_mask_;
    }
    return ret;
}