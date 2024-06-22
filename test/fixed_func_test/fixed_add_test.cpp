#include "ezpc_scilib/ezpc_utils.h"
#include <model.h>
#include <cmath>
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
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << random_share[i] << " ";
    // }

    // std::cout << "\n";
    // std::cout << "mask_x:" << mask_x << " \n";
    // for (int i = 0; i < length; i++)
    // {
    //     random_share[i] &= mask_x;
    // }
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << random_share[i] << " ";
    // }
    // std::cout << "\n";

    BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));

    BFVKey *alice = new BFVKey(sci::ALICE, bfv_parm);

    sci::OTPack *otpack;
    sci::IOPack *iopack;
    MillionaireProtocol *mill;

    sci::PRG128 prg2;
    size_t size = 5;
    uint64_t *x = new uint64_t[size];
    prg2.random_data(x, size * sizeof(uint64_t));

    // std::cout << "no signed:";
    // for (int i = 0; i < size; i++)
    // {
    //     std::cout << x[i] << " ";
    // }
    // std::cout << "\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    double *input = new double(5);
    int64_t *fix_input = new int64_t(5);
    uint64_t *unsig_fix_input = new uint64_t(5);
    for (size_t i = 0; i < 5; i++)
    {
        input[i] = dist(gen);
        fix_input[i] = static_cast<int64_t>(input[i] * (1ULL << 12));  // (-5, 5)
        unsig_fix_input[i] = sci::neg_mod(fix_input[i], (1ULL << 37)); // (0, 10)
        std::cout << input[i] << " ";
        std::cout << fix_input[i] << " ";
        std::cout << unsig_fix_input[i] << " ";
        std::cout << "\n";
    }

    FixOp *fix = new FixOp(sci::PUBLIC, iopack, otpack);
    FixArray input1 = fix->input(sci::PUBLIC, 5, unsig_fix_input, true, 37, 12);

    FixArray result(input1.party, 1, input1.signed_, input1.ell, input1.s);
    double div_colum = 1.0 / 5.0;
    std::cout << "div_colum : " << div_colum << "\n";
    const uint64_t fix_column = sci::neg_mod(static_cast<int64_t>(div_colum * (1ULL << 12)), (1ULL << 37));
    // FixArray div(sci::PUBLIC, 1, true, 37, 12);
    std::cout << "fix_column : " << fix_column << "\n";
    size_t i, j;
    print_fix(input1);

    FixArray tmp_input(input1.party, 1, input1.signed_, input1.ell, input1.s);
    FixArray tmp_output(input1.party, 1, input1.signed_, input1.ell, input1.s);
    for (i = 0; i < 5; i++)
    {
        tmp_input.data[0] = input1.data[i];
        // fix->add();
    }

    // #pragma omp parallel for
    for (i = 0; i < 5; i++)
    {
        result.data[0] += input1.data[i];
        std::cout << "input1 " << input1.data[i] << " \n";
    }
    result.data[0] = (result.data[0] & mask_x);
    std::cout << "sum: " << result.data[0] << "\n";
    print_fix(result);
    result = fix->mul(result, fix_column);
    // result = fix->div(result, div); // TODO: MathFunction::DIV

    print_fix(result);

    delete random_share;
    delete bfv_parm;
    delete alice;
    delete otpack;
    delete iopack;
    delete mill;
    delete x;
    delete fix_input;
    delete input;
    delete unsig_fix_input;
    delete fix;
}

// const size_t bfv_poly_modulus_degree = 8192;
// const size_t bfv_slot_count = bfv_poly_modulus_degree;
// const vector<int> bfv_coeff_bit_sizes = {54, 54, 55, 55};
// const int32_t bitlength = 29;
// const uint64_t bfv_plain_mod = default_prime_mod.at(bitlength);
