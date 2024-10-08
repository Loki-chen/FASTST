#ifndef FIXED_POINT_MATH_H__
#define FIXED_POINT_MATH_H__
#include "BuildingBlocks/linear-ot.h"
#include "Math/math-functions.h"
// #include "fixed-math-coeffs.h"
#include "fixed-point.h"
#include <cmath>
class FPMath
{
public:
    int party;
    sci::IOPack *iopack;
    sci::OTPack *otpack;
    BoolOp *bool_op;
    FixOp *fix;
    MathFunctions *math;

    FPMath(int party, sci::IOPack *iopack, sci::OTPack *otpack)
    {
        this->party = party;
        this->iopack = iopack;
        this->otpack = otpack;
        this->bool_op = new BoolOp(party, iopack, otpack);
        this->fix = new FixOp(party, iopack, otpack);
        this->math = new MathFunctions(party, iopack, otpack);
    }

    ~FPMath()
    {
        delete bool_op;
        delete fix;
        delete math;
    }

    // Fixed-Point Math Functions: returns OP(x[i]), OP = {sinpi, cospi, tanpi, exp2, log2, exp, ln, erf}
    // x must be secret-shared

    std::tuple<FixArray, FixArray> exp4(const FixArray &x);

    FixArray lookup_table_exp(const FixArray &x);
    FixArray tanh_inner(const FixArray &x);
    FixArray tanh_inner_preprocess(const FixArray &x);
    FixArray tanh_approx(const FixArray &x);
    FixArray gt_p_sub(const FixArray &x, const FixArray &p);
    FixArray location_gt_p_sub(const FixArray &x, const FixArray &p);
    FixArray sqrt_(const FixArray &x, bool recp_sqrt);
    std::tuple<FixArray, FixArray, FixArray> bitonic_sort_and_swap(const FixArray &x, FixArray softmax_v_, FixArray h1_,
                                                                   bool swap);
    void print(const FixArray &x);
    vector<FixArray> mean(const vector<FixArray> &x);
    vector<FixArray> standard_deviation(const vector<FixArray> &x, const vector<FixArray> mean);
    double sqrt_(float x);
    int64_t LUT_neg_exp(int64_t val_in, int32_t s_in, int32_t s_out);
    FixArray location_exp(const FixArray &x, int scale_in, int scale_out);
    inline int64_t fpSaturate(int32_t inp) { return (int64_t)inp; }
    FixArray dot(const FixArray &x, const FixArray &y, size_t dim1, size_t dim2, size_t dim3, int ell,
                 bool trans = false, uint8_t *msb_x = nullptr, uint8_t *msb_y = nullptr);
    FixArray zero_sum_modP(size_t row, size_t column, uint64_t prime_mod, int ell, int scale);

    // BOLT
    FixArray gelu_bolt(const FixArray &x);
    std::tuple<vector<FixArray>, FixArray> softmax_bolt(const vector<FixArray> &x);
    vector<FixArray> layer_norm_bolt(const vector<FixArray> &x, FixArray &w, FixArray &b);
    // IRON
    FixArray gelu_iron(const FixArray &x);
    vector<FixArray> softmax_iron(const vector<FixArray> &x);
    vector<FixArray> layer_norm_iron(const vector<FixArray> &x, FixArray &w, FixArray &b);
};

#endif // FIXED_POINT_MATH_H__