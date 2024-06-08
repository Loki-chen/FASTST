#include <model.h>
#include "ezpc_scilib/ezpc_utils.h"
#define TEST

int main()
{
    int NL_ELL = 37;
    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));
    // uint64_t ell_mask = (1ULL << 37) - 1; // 2 ** 37 -1

    int length = 2 * 50;
    uint64_t *random_share = new uint64_t[length];
    sci::PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));
    for (int i = 0; i < 10; i++)
    {
        std::cout << random_share[i] << " ";
    }

    std::cout << "\n";
    std::cout << "mask_x:" << mask_x << " \n";
    for (int i = 0; i < length; i++)
    {
        random_share[i] &= mask_x;
    }
    for (int i = 0; i < 10; i++)
    {
        std::cout << random_share[i] << " ";
    }

    BFVparm *bfv_parm = new BFVparm(sci::ALICE, 8192, {54, 54, 55, 55}, default_prime_mod.at(29));

    BFVKey *alice = new BFVKey(bfv_parm->party, bfv_parm->context);

    sci::OTPack *otpack;

    MillionaireProtocol *mill;

    sci::PRG128 prg2;
    uint64_t *x = new uint64_t[40];
    prg2.random_mod_p<uint64_t>(x, 40 * sizeof(uint64_t), 64);

    FixArray *fix;
}

// const size_t bfv_poly_modulus_degree = 8192;
// const size_t bfv_slot_count = bfv_poly_modulus_degree;
// const vector<int> bfv_coeff_bit_sizes = {54, 54, 55, 55};
// const int32_t bitlength = 29;
// const uint64_t bfv_plain_mod = default_prime_mod.at(bitlength);