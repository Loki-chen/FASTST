#include "fixed-attention.h"
#include "model.h"

Fixed_Attention::Fixed_Attention(int layer, BFVKey *party, BFVParm *parm,
                                 sci::NetIO, FPMath *fpmath, FPMath *fpmath_public, Conversion *conv, int head_)
    : FixedProtocol(layer, party, parm, io, fpmath, fpmath_public, conv), , head(head_) {}

bfv_matrix Fixed_Attention::forward(const bfv_matrix &input) const
{

    sci::PRG128 prg;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);

    uint64_t *x = new uint64_t[input.size()];
    for (size_t i = 0; i < input.size(); i++)
    {
        x[i] = input[i];
    }

    if (party->party == sci::ALICE)
    {
        double ra = dist(gen);

        uint64_t *ra_xa_wa = new uint64_t[d_module * d_k];
        for (size_t i = 0; i < count; i++)
        {
            /* code */
        }

        FixArray fix_ra = fpmath->fix->input(sci::ALICE, batch_size * d_module,
                                             (sci::neg_mod(static_cast<int64_t>(ra * (1ULL << (DEFAULT_SCALE))), (1ULL << DEFAULT_ELL))),
                                             true, DEFAULT_ELL, DEFAULT_SCALE);
        FixArray fix_xa = fpmath->fix->input(sci::ALICE, batch_size * d_module, x, true, DEFAULT_ELL, DEFAULT_SCALE);

        fix_ra.party = sci::PUBLIC; // just to make the mul useful.
        FixArray fix_ra_xa = fpmath->fix->mul(fix_xa, fix_ra, DEFAULT_ELL);
        fix_ra_xa = fpmath->fix->location_truncation(fix_ra_xa, DEFAULT_SCALE);
        // x_a_ra_wa, x_a_ra_qa, x_a_ra_va

        // Alice End
    }
    else
    {
    }
}