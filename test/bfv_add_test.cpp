#include <model.h>
#include "ezpc_scilib/fixed-point.h"
#define TEST

inline uint64_t Saturate(uint32_t inp) { return (uint64_t)inp; }

void cleartext_MatSub(uint64_t *A, const uint64_t *B, u_int64_t *C, uint32_t I,
                      uint32_t J, uint32_t shrA, uint32_t shrB, uint32_t shrC,
                      int32_t demote)
{
    uint32_t shiftA = log2(shrA);
    uint32_t shiftB = log2(shrB);
    uint32_t shiftC = log2(shrC);
    uint32_t shift_demote = log2(demote);
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            uint64_t a = (uint64_t)A[i * J + j];
            uint64_t b = (uint64_t)B[i * J + j];

#ifdef DIV_RESCALING
            a = a / (shrA * shrC);
            b = b / (shrB * shrC);
#else
            a = a >> (shiftA + shiftC);
            b = b >> (shiftB + shiftC);
#endif

            uint64_t c = a - b;

#ifdef DIV_RESCALING
            C[i * J + j] = Saturate(c / demote);
#else
            C[i * J + j] = Saturate(c >> shift_demote);
#endif
        }
    }
    return;
}

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
}