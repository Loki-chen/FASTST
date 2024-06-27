#include <iostream>
#include <vector>
#include "ezpc_scilib/ezpc_utils.h"
#include "utils.h"
#define TEST

int main(int argc, const char **argv)
{
    if (argc > 1)
    {
        int party_ = argv[1][0] - '0';
        assert(party_ == sci::ALICE || party_ == sci::BOB);
        if (party_ == sci::ALICE)
        {
            std::cout << "Party: ALICE"
                      << "\n";
        }
        else if (party_ == sci::BOB)
        {
            std::cout << "Party: BOB"
                      << "\n";
        }
        int s = 13;
        int ell = 37;
        int64_t plain_mod = 536903681;

        int array_size = 5;
        int len = 1;

        sci::PRG128 prg;

        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);

        sci::IOPack *iopack = new sci::IOPack(party_, 56789);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        FPMath *fpmath = new FPMath(party_, iopack, otpack);

        if (party_ == sci::ALICE)
        {
            Conversion *conv = new Conversion();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-1, 1);
            double *input = new double[len * array_size];
            int64_t *fix_input = new int64_t[len * array_size];
            uint64_t *unsig_fix_input = new uint64_t[len * array_size];
            for (size_t j = 0; j < len; j++)
            {
                for (size_t i = 0; i < array_size; i++)
                {
                    input[j * array_size + i] = dist(gen);
                    fix_input[j * array_size + i] = static_cast<int64_t>(input[j * array_size + i] * (1ULL << s));             // (-5, 5)
                    unsig_fix_input[j * array_size + i] = sci::neg_mod(fix_input[j * array_size + i], (int64_t)(1ULL << ell)); // (0, 10)
                    std::cout << unsig_fix_input[j * array_size + i] << " ";
                }
            }

            uint64_t *output = new uint64_t[len * array_size];
            output = conv->Ring_to_Prime(unsig_fix_input, len * array_size, ell, plain_mod);

            for (size_t j = 0; j < len * array_size; j++)
            {
                std::cout << sci::signed_val(output[j], plain_mod) << " ";
            }
            std::cout << "\n";

            delete[] output;
            delete[] input;
            delete[] fix_input;
            delete[] unsig_fix_input;
        }
        Conversion *conv_party = new Conversion();
        // test conversion print--to--ring
        uint64_t *input_pring = new uint64_t[len * array_size];
        prg.random_mod_p<uint64_t>(input_pring, len * array_size, plain_mod);

        input_pring = conv_party->Prime_to_Ring(party_, input_pring, len * array_size, ell, plain_mod, s, s, fpmath);

        for (size_t j = 0; j < len * array_size; j++)
        {
            std::cout << sci::signed_val(input_pring[j], ell) << " ";
        }
        std::cout << "\n";

        delete[] input_pring;
        return 0;
    }
}