#include "conversion.h"

void Prime_to_Ring(BFVLongCiphertext &ciphertext, uint64_t *share, Evaluator *evaluator) {}

uint64_t *Conversion::Ring_to_Prime(uint64_t *input, int length, int ell, int plain_mod)
{
#ifdef LOG
    auto t_conversion = high_resolution_clock::now();
#endif

    uint64_t *output = new uint64_t[length];
    vector<uint64_t> tmp;
    for (size_t i = 0; i < length; i++)
    {
        tmp[i] = sci::neg_mod(sci::signed_val(input[i], ell), (int64_t)plain_mod);
    }
    memcpy(output, &tmp, length * uint64_t);
#ifdef LOG
    t_total_conversion += interval(t_conversion);
#endif
    return output;
}