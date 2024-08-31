#include "fixed-math.h"

using namespace std;
using namespace sci;

#define FRAC_RANGE 9
#define FP_INTMD_M_BITS 27
#define FP_INTMD_E_BITS 8
#define PI_DOUBLE 3.1415926535897932384626433832795028841971693993751058209749445923078164062
#define LOG2E 1.44269504088896340735992468100189213742664595415298593413544940693110921918118507988552662289350634449699
#define LOGE2 0.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687
#define TWO_INV_SQRT_PI 1.128379167095512573896158903121545171688101258657997713688171443421284936882
#define NEG_LOGE2_INV \
    1.442695040888963423535598661526235116567603930130965898132921686199121361438241594804331289503955470328942932380383923264

FixArray get_idx_from_input(FixOp *fix, const FixArray &delta_m, const FixArray &delta_e, int idx_m_bits,
                            int idx_e_bits, int e_offset)
{
    assert(delta_m.party != PUBLIC && delta_e.party != PUBLIC);
    assert(delta_m.size == delta_e.size);
    assert(idx_m_bits + idx_e_bits <= delta_e.ell);
    FixArray idx_hi = fix->reduce(fix->add(delta_e, e_offset), idx_m_bits + idx_e_bits);
    idx_hi.signed_ = false;
    if (idx_m_bits == 0)
    {
        return idx_hi;
    }
    idx_hi = fix->mul(idx_hi, 1 << idx_m_bits, idx_m_bits + idx_e_bits);
    FixArray idx_lo = fix->truncate_reduce(delta_m, delta_m.ell - 1 - idx_m_bits);
    idx_lo = fix->sub(idx_lo, 1 << idx_m_bits);
    if (idx_m_bits + idx_e_bits < idx_m_bits + 1)
    {
        idx_lo = fix->reduce(idx_lo, idx_m_bits + idx_e_bits);
    }
    idx_lo.s = 0;
    BoolArray all_0 = fix->bool_op->input(ALICE, delta_m.size, uint8_t(0));
    FixArray idx = fix->add(idx_hi, fix->extend(idx_lo, idx_m_bits + idx_e_bits, all_0.data));
    return idx;
}

std::tuple<FixArray, FixArray> FPMath::exp4(const FixArray &x)
{

    /*
    l = np.floor((x / -math.log(2)))
    p = x + l*math.log(2)
    fp = poly(p)
    return fp / (2**l)
    */

    // print_fix(x);

    int ell = x.ell;
    int scale = x.s;

    // All 0 and all 1 array for msb arg
    BoolArray all_0 = bool_op->input(ALICE, x.size, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, x.size, 1);

    // ln2
    FixArray ln2 = fix->input(PUBLIC, x.size, uint64_t(2839), true, ell, scale);
    // print_fix(ln2);

    // inverse of negative ln2
    FixArray inl = fix->input(PUBLIC, x.size, uint64_t(-5909), true, ell, scale);
    // print_fix(inl);

    // x / -math.log(2)
    // Truncate to original scale and bitlength
    FixArray x_inl = fix->mul(x, inl, ell + scale);
    // Optimization: local truncation
    x_inl = fix->truncate_reduce(x_inl, scale);
    // x_inl =  fix->reduce(x_inl, ell);
    // print_fix(x_inl);

    // Get the integer part and scale back
    FixArray l_short = fix->truncate_reduce(x_inl, scale);
    FixArray l_short_raw = l_short;
    FixArray l = fix->scale_up(l_short, ell, scale);

    // l*math.log(2)
    FixArray l_ln2 = fix->mul(l, ln2, ell + scale, all_0.data, all_0.data);
    l_ln2 = fix->truncate_reduce(l_ln2, scale);
    // l_ln2 =  fix->reduce(l_ln2, ell);

    // Get the decimal part
    FixArray p = fix->add(x, l_ln2);
    // Optimization: We don't need that much bit as p \in (-ln2, 0])
    p = fix->reduce(p, scale + 2);

    // Polynomial fit
    FixArray poly_p = fix->poly1(p);
    poly_p = fix->extend(poly_p, ell, all_0.data);

    l_short.signed_ = false;
    // Optimization: The polynomial result is within [0, ~0.7)
    // Thus the upper bound of shift is scale + 1

    FixArray bound = fix->input(PUBLIC, l_short.size, 13, false, l_short.ell, 0);
    BoolArray gt_bound = fix->GT(l_short, bound);
    l_short = fix->if_else(gt_bound, bound, l_short);

    FixArray ret = fix->right_shift(poly_p, l_short, scale + 1, all_1.data);

    return make_tuple(ret, l_short_raw);
}

FixArray FPMath::lookup_table_exp(const FixArray &x)
{
    FixArray ret(party, x.size, x.signed_, x.ell, x.s);
    math->lookup_table_exp(x.size, x.data, ret.data, x.ell, x.ell, x.s, x.s);
    return ret;
}

FixArray FPMath::tanh_inner_preprocess(const FixArray &x)
{
    int N = x.size;
    int ell = x.ell;
    int s = x.s;

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N, 1);

    // Const
    FixArray t0 = fix->input(PUBLIC, x.size, uint64_t((-4.259314087994767) * (1 << s)), true, ell, s);
    FixArray t1 = fix->input(PUBLIC, x.size, uint64_t((18.86353816972803) * (1 << s)), true, ell, s);
    FixArray t2 = fix->input(PUBLIC, x.size, uint64_t((-36.42402897526823) * (1 << s)), true, ell, s);
    FixArray t3 = fix->input(PUBLIC, x.size, uint64_t((-0.013232131886235352) * (1 << s)), true, ell, s);
    FixArray t4 = fix->input(PUBLIC, x.size, uint64_t((-3.3289339650097993) * (1 << s)), true, ell, s);
    FixArray t5 = fix->input(PUBLIC, x.size, uint64_t((-0.0024920889620412097) * (1 << s)), true, ell, s);

    // p1(x) = (x + t0)*x + t1
    // Range: >0
    FixArray p1 = fix->add(x, t0);
    p1 = fix->mul(p1, x, ell + s, all_0.data, all_0.data);
    p1 = fix->truncate_reduce(p1, s);
    p1 = fix->add(p1, t1);

    // p2(x) = (p1(x) + x + t2)*p1(x)*x*t3 + t4*x + t5

    // (p1(x) + x + t2) < 0
    FixArray p2 = fix->add(p1, x);
    p2 = fix->add(p2, t2);

    // (p1(x) + x + t2)*p1(x) < 0
    p2 = fix->mul(p2, p1, ell + s, all_1.data, all_0.data);
    p2 = fix->truncate_reduce(p2, s);

    // p2(x) = (p1(x) + x + t2)*p1(x)*t3 >0
    p2 = fix->mul(p2, t3, ell + s, all_1.data, all_1.data);
    p2 = fix->truncate_reduce(p2, s);

    // p2(x) = (p1(x) + x + t2)*p1(x)*x*t3 < 0
    p2 = fix->mul(p2, x, ell + s, all_1.data, all_0.data);
    p2 = fix->truncate_reduce(p2, s);

    FixArray t4x = fix->mul(x, t4, ell + s, all_0.data, all_1.data);
    t4x = fix->truncate_reduce(t4x, s);

    p2 = fix->add(p2, t4x);
    p2 = fix->add(p2, t5);

    return p2;
}

FixArray FPMath::tanh_inner(const FixArray &x)
{
    int N = x.size;
    int ell = x.ell;
    int s = x.s;

    // Const
    FixArray a = fix->input(PUBLIC, x.size, uint64_t((-0.013232131886235352) * (1 << s)), true, ell, s);
    FixArray b = fix->input(PUBLIC, x.size, uint64_t((0.09948747962825866) * (1 << s)), true, ell, s);
    FixArray c = fix->input(PUBLIC, x.size, uint64_t((-0.20093640347818847) * (1 << s)), true, ell, s);
    FixArray d = fix->input(PUBLIC, x.size, uint64_t((-0.17616532856475706) * (1 << s)), true, ell, s);
    FixArray e = fix->input(PUBLIC, x.size, uint64_t((1.0542492677156243) * (1 << s)), true, ell, s);
    FixArray f = fix->input(PUBLIC, x.size, uint64_t((-0.0024920889620412097) * (1 << s)), true, ell, s);

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N, 1);

    //
    FixArray x_square = fix->mul(x, x, ell + s, all_0.data, all_0.data);
    x_square = fix->truncate_reduce(x_square, s);

    FixArray x_cube = fix->mul(x_square, x, ell + s, all_0.data, all_0.data);
    x_cube = fix->truncate_reduce(x_cube, s);

    FixArray x_four = fix->mul(x_square, x_square, ell + s, all_0.data, all_0.data);
    x_four = fix->truncate_reduce(x_four, s);

    FixArray x_five = fix->mul(x_four, x, ell + s, all_0.data, all_0.data);
    x_five = fix->truncate_reduce(x_five, s);

    FixArray x_five_a = fix->mul(x_five, a, ell + s, all_0.data, all_1.data);
    x_five_a = fix->truncate_reduce(x_five_a, s);

    FixArray x_four_b = fix->mul(x_four, b, ell + s, all_0.data, all_0.data);
    x_four_b = fix->truncate_reduce(x_four_b, s);

    FixArray x_cube_c = fix->mul(x_cube, c, ell + s, all_0.data, all_1.data);
    x_cube_c = fix->truncate_reduce(x_cube_c, s);

    FixArray x_square_d = fix->mul(x_square, d, ell + s, all_0.data, all_1.data);
    x_square_d = fix->truncate_reduce(x_square_d, s);

    FixArray x_e = fix->mul(x, e, ell + s, all_0.data, all_0.data);
    x_e = fix->truncate_reduce(x_e, s);

    f = fix->add(f, x_e);
    f = fix->add(f, x_square_d);
    f = fix->add(f, x_cube_c);
    f = fix->add(f, x_four_b);
    f = fix->add(f, x_five_a);

    return f;
}

FixArray FPMath::tanh_approx(const FixArray &x)
{
    int N = x.size;
    int ell = x.ell;
    int s = x.s;

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N, 1);

    FixArray cons_2 = fix->input(PUBLIC, x.size, uint64_t((2 << s)), true, ell, s);
    FixArray cons_1 = fix->input(PUBLIC, x.size, uint64_t((1 << s)), true, ell, s);
    FixArray cons_neg_1 = fix->input(PUBLIC, x.size, uint64_t((-1 << s)), true, ell, s);

    BoolArray pos = fix->GT(x, 0);
    FixArray neg_x = fix->mul(x, -1);
    FixArray abs_x = fix->if_else(pos, x, neg_x);

    FixArray cond_fix = fix->B2A(pos, true, ell);
    cond_fix = fix->scale_up(cond_fix, ell, s);
    FixArray sign_x = fix->mul(cond_fix, cons_2, ell + s, all_0.data, all_0.data);
    sign_x = fix->truncate_reduce(sign_x, s);
    sign_x = fix->add(sign_x, cons_neg_1);

    BoolArray gt3 = fix->GT(abs_x, (uint64_t)(2.855 * (1 << s)));
    FixArray abs_tanh = fix->if_else(gt3, cons_1, tanh_inner_preprocess(abs_x));
    FixArray ret = fix->mul(abs_tanh, sign_x, ell + s, all_0.data);
    ret = fix->truncate_reduce(ret, s);

    return ret;
}

FixArray FPMath::sqrt_(const FixArray &x, bool recp_sqrt)
{
    FixArray ret(party, x.size, x.signed_, x.ell, x.s);
    math->sqrt(x.size, x.data, ret.data, x.ell, x.ell, x.s, x.s, recp_sqrt);
    return ret;
}

FixArray FPMath::gt_p_sub(const FixArray &x, const FixArray &p)
{
    BoolArray gt = fix->GT(x, p);
    FixArray sub = fix->sub(x, p);
    return fix->if_else(gt, sub, x);
}

FixArray FPMath::location_gt_p_sub(const FixArray &x, const FixArray &p)
{
    BoolArray gt = fix->location_GT(x, p);
    FixArray sub = fix->sub(x, p);
    return fix->location_if_else(gt, sub, x); // use location_if_else, without mul multiplexer.
}

void FPMath::print(const FixArray &x) { print_fix(x); }

BoolArray bitonic_reverse(const BoolArray &x, int array_size, int cur_depth)
{
    BoolArray ret(x.party, x.size);
    int block_size = 2 * cur_depth;
    int num_block = array_size / block_size;
    for (int i = 0; i < num_block; i++)
    {

        for (int j = 0; j < cur_depth; j++)
        {
            int index = i * cur_depth + j;
            if (i % 2 == 1)
            {
                ret.data[index] = x.data[index] ^ ((x.party != BOB) ? 1 : 0);
            }
            else
            {
                ret.data[index] = x.data[index];
            }
        }
    }
    return ret;
}

tuple<FixArray, FixArray, FixArray> FPMath::bitonic_sort_and_swap(const FixArray &x_, FixArray softmax_v_, FixArray h1_,
                                                                  bool swap)
{
    FixArray x = x_;
    FixArray softmax_v = softmax_v_;
    FixArray h1 = h1_;

    int array_size = x.size;
    int max_depth = array_size / 2;
    int cur_depth = 1;

    int common_dim;

    while (cur_depth <= max_depth)
    {
        int cur_iter = cur_depth;
        while (cur_iter > 0)
        {
            int block_size = 2 * cur_iter;
            int num_block = array_size / block_size;

            vector<int> index_left;
            vector<int> index_right;

            FixArray array_left(party, x.size / 2, x.signed_, x.ell, x.s);
            FixArray array_right(party, x.size / 2, x.signed_, x.ell, x.s);
            FixArray array_reverse(party, x.size, x.signed_, x.ell, x.s);

            FixArray softmax_v_reverse;
            FixArray h1_reverse;

            if (swap)
            {
                common_dim = softmax_v.size / x.size;
                assert(common_dim == 768);

                softmax_v_reverse =
                    fix->input(party, softmax_v.size, (uint64_t)0, softmax_v.signed_, softmax_v.ell, softmax_v.s);
                h1_reverse = fix->input(party, h1.size, (uint64_t)0, h1.signed_, h1.ell, h1.s);

                for (int i = 0; i < num_block; i++)
                {
                    for (int j = 0; j < cur_iter; j++)
                    {
                        int pos_x = i * block_size + j;
                        int pos_y = i * block_size + j + cur_iter;
                        index_left.push_back(pos_x);
                        index_right.push_back(pos_y);
                        array_reverse.data[pos_x] = x.data[pos_y];
                        array_reverse.data[pos_y] = x.data[pos_x];

                        memcpy(&softmax_v_reverse.data[pos_x * common_dim], &softmax_v.data[pos_y * common_dim],
                               common_dim * sizeof(uint64_t));

                        memcpy(&softmax_v_reverse.data[pos_y * common_dim], &softmax_v.data[pos_x * common_dim],
                               common_dim * sizeof(uint64_t));

                        memcpy(&h1_reverse.data[pos_x * common_dim], &h1.data[pos_y * common_dim],
                               common_dim * sizeof(uint64_t));

                        memcpy(&h1_reverse.data[pos_y * common_dim], &h1.data[pos_x * common_dim],
                               common_dim * sizeof(uint64_t));
                    }
                }
            }
            else
            {
                for (int i = 0; i < num_block; i++)
                {
                    for (int j = 0; j < cur_iter; j++)
                    {
                        index_left.push_back(i * block_size + j);
                        index_right.push_back(i * block_size + j + cur_iter);
                        array_reverse.data[i * block_size + j] = x.data[i * block_size + j + cur_iter];
                        array_reverse.data[i * block_size + j + cur_iter] = x.data[i * block_size + j];
                    }
                }
            }

            for (int i = 0; i < array_size / 2; i++)
            {
                array_left.data[i] = x.data[index_left[i]];
                array_right.data[i] = x.data[index_right[i]];
            }

            // print_fix(array_left);
            // print_fix(array_right);
            // print_fix(array_reverse);

            BoolArray lt = fix->LT(array_left, array_right);
            BoolArray cmp_extend = BoolArray(party, lt.size * 2);

            // Reverse some comparisons
            BoolArray cmp = bitonic_reverse(lt, array_size, cur_depth);
            for (int i = 0; i < num_block; i++)
            {
                for (int j = 0; j < cur_iter; j++)
                {
                    cmp_extend.data[i * block_size + j] = cmp.data[i * cur_iter + j];
                    cmp_extend.data[i * block_size + j + cur_iter] = cmp.data[i * cur_iter + j];
                }
            }
            // print_bool(lt);
            // print_bool(cmp);
            // print_bool(cmp_extend);
            // assert(0);

            // print_fix(x);
            x = fix->if_else(cmp_extend, x, array_reverse);
            // print_fix(x);

            if (swap)
            {
                BoolArray cmp_flat = BoolArray(party, cmp_extend.size * common_dim);
                for (int i = 0; i < cmp_flat.size; i++)
                {
                    cmp_flat.data[i] = cmp_extend.data[i / common_dim];
                }

                softmax_v = fix->if_else(cmp_flat, softmax_v, softmax_v_reverse);
                h1 = fix->if_else(cmp_flat, h1, h1_reverse);
            }

            cur_iter /= 2;
        }
        cur_depth *= 2;
    }

    return make_tuple(x, softmax_v, h1);
}

double FPMath::sqrt_(float x) { return std::sqrt(x); }

// math function for FASTLMPI
vector<FixArray> FPMath::mean(const vector<FixArray> &x)
{
    int party_origin = x[0].party;
    int N = x.size();
    int n = x[0].size;
    int ell = x[0].ell;
    int s = x[0].s;
    bool signed_ = x[0].signed_;
    FixArray sum_res = fix->tree_sum(x);
    uint64_t dn = static_cast<uint64_t>((1.0 / n) * (1ULL << s));
    FixArray fix_dn = fix->input(sci::PUBLIC, N, dn, true, ell, s);
    sum_res.party = sci::ALICE;
    FixArray avg = fix->mul(sum_res, fix_dn, ell);
    avg.party = sci::PUBLIC;
    avg = fix->location_truncation(avg, s);
    avg.party = party_origin;
    vector<FixArray> ret(N);
    for (int i = 0; i < N; i++)
    {
        ret[i] = FixArray(party_origin, 1, signed_, ell, s);
        memcpy(ret[i].data, &avg.data[i], sizeof(uint64_t));
    }

    return ret;
}

vector<FixArray> FPMath::standard_deviation(const vector<FixArray> &x, const vector<FixArray> mean)
{
    int party_origin = x[0].party;
    int N = x.size();
    int n = x[0].size;
    int ell = x[0].ell;
    int s = x[0].s;
    bool signed_ = x[0].signed_;

    uint64_t dn = uint64_t(((1.0 / n) * pow(2, s)));
    FixArray fix_dn = fix->input(sci::PUBLIC, N, dn, true, ell, s);
    vector<FixArray> tmp_ret(N);
    vector<FixArray> tmp_y(N);
    for (size_t i = 0; i < N; i++)
    {
        tmp_ret[i] = fix->sub(x[i], mean[i].data[0]);
        tmp_ret[i] = fix->public_mul(tmp_ret[i], tmp_ret[i], ell + 2 * s);
        tmp_ret[i] = fix->location_truncation(tmp_ret[i], s);
    }

    FixArray sum_res = fix->tree_sum(tmp_ret); // obtain (((g)^s)^2delta)^2
    sum_res.party = sci::ALICE;
    FixArray avg = fix->mul(sum_res, fix_dn, ell);
    avg.party = sci::PUBLIC;
    avg = fix->location_truncation(avg, s);
    uint64_t unsig_fix_delta;
    vector<FixArray> ret(N);
    for (size_t i = 0; i < N; i++)
    {
        ret[i] = FixArray(party_origin, 1, signed_, ell, s);
        double delta = double(avg.data[i]) / (1ULL << s);
        unsig_fix_delta = static_cast<int64_t>(1.0 / sqrt_(float(delta)) * (1ULL << s)); // (-5, 5)
        unsig_fix_delta = sci::neg_mod(unsig_fix_delta, (int64_t)(1ULL << ell));         // (0, 10)
        ret[i].data[0] = unsig_fix_delta;
    }
    return ret;
}

int64_t FPMath::LUT_neg_exp(int64_t val_in, int32_t s_in, int32_t s_out)
{
    if (s_in < 0)
    {
        s_in *= -1;
        val_in *= (1 << (s_in));
        s_in = 0;
    }
    int64_t res_val = exp(-1.0 * (val_in / double(1LL << s_in))) * (1LL << s_out);
    return res_val;
}

FixArray FPMath::location_exp(const FixArray &x, int scale_in, int scale_out)
{
    assert(x.party == PUBLIC);
    int digit_limit = 8;
    BoolArray msb_x = fix->MSB(x, x.ell);
    FixArray neg_x(x.party, x.size, true, x.ell, x.s);
    int64_t *A = new int64_t[x.size];
    for (size_t i = 0; i < x.size; i++) // FIX:: use location_if_else
    {
        if (msb_x.data[i])
        {
            neg_x.data[i] = x.data[i];
        }
        else
        {
            neg_x.data[i] = x.data[i] * -1;
        }
    }
    vector<double> neg_x_double = neg_x.get_native_type<double>();

    for (size_t i = 0; i < x.size; i++)
    {
        A[i] = neg_x_double[i] * (1ULL << x.s);
    }

    int32_t s_demote = log2(1);
    int num_digits = ceil(double(x.ell) / digit_limit);
    int last_digit_size = x.ell - (num_digits - 1) * digit_limit;
    int64_t digit_mask = (digit_limit == 64 ? -1 : (1LL << digit_limit) - 1);
    int64_t A_digits[num_digits];
    FixArray ret(x.party, x.size, true, x.ell, x.s);

    int64_t error = 0;
    for (size_t i = 0; i < x.size; i++)
    {
        assert(A[i] <= 0);
        int64_t neg_A = -1LL * (A[i]);
        for (int j = 0; j < digit_limit; j++)
        {
            A_digits[j] = (neg_A >> (j * digit_limit)) & digit_mask;
            A_digits[j] = this->LUT_neg_exp(A_digits[j], scale_in - digit_limit * j, scale_out);
        }
        for (int j = 1; j < digit_limit; j *= 2)
        {
            for (int k = 0; k < digit_limit and k + j < digit_limit; k += 2 * j)
            {
                A_digits[k] = (A_digits[k + j] * A_digits[k]) >> scale_out;
            }
        }
        ret.data[i] = this->fpSaturate(A_digits[0] >> s_demote);
    }
    vector<double> real_ret = ret.get_native_type<double>();
    uint64_t *tmp = new uint64_t[ret.size];

    for (size_t i = 0; i < x.size; i++)
    {
        if (!msb_x.data[i]) // if = 1, x have x.signed =-, else x.signed= +
        {
            real_ret[i] = 1.0 / real_ret[i];
        }
        tmp[i] = sci::neg_mod(static_cast<int64_t>(real_ret[i] * (1ULL << scale_out)), (int64_t)(1ULL << ret.ell));
    }
    memcpy(ret.data, tmp, ret.size * sizeof(uint64_t));

    delete[] A;
    delete[] tmp;
    return ret;
}

FixArray FPMath::dot(const FixArray &x, const FixArray &y, size_t dim1, size_t dim2, size_t dim3, int ell, bool trans,
                     uint8_t *msb_x, uint8_t *msb_y)
{
    assert(x.signed_ || (x.signed_ == y.signed_));
    assert(ell >= x.ell && ell >= y.ell && ell <= x.ell + y.ell);
    assert(ell < 64);
    assert(x.size == dim1 * dim2 && y.size == dim2 * dim3);

    FixArray ret(this->party, dim1 * dim3, x.signed_, ell, x.s + y.s);
    if (x.party == PUBLIC || y.party == PUBLIC || x.party == y.party)
    {
        FixArray x_ext = fix->extend(x, ell, msb_x);
        FixArray y_ext = fix->extend(y, ell, msb_y);
        uint64_t ret_mask = ret.ell_mask();
        if (!trans)
        {
#pragma omp parallel for
            for (size_t i = 0; i < dim1; i++)
            {
                const size_t base_idx1 = i * dim2;
                const size_t base_idx2 = i * dim3;
                for (size_t k = 0; k < dim2; k++)
                {
                    const size_t base_idx3 = k * dim3;
                    const auto tmp = x_ext.data[base_idx1 + k];
                    for (size_t j = 0; j < dim3; j++)
                    {
                        ret.data[base_idx2 + j] += ((tmp * y_ext.data[base_idx3 + j]) & ret_mask);
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (size_t i = 0; i < dim1; i++)
            {
                const size_t base_idx1 = i * dim2;
                const size_t base_idx2 = i * dim3;
                for (size_t j = 0; j < dim3; j++)
                {
                    const size_t base_idx3 = j * dim2;
                    uint64_t sum = 0;
                    for (size_t k = 0; k < dim2; k++)
                    {
                        sum += ((x_ext.data[base_idx1 + k] * y_ext.data[base_idx3 + k]) & ret_mask);
                    }
                    ret.data[base_idx2 + j] = sum;
                }
            }
        }
        fix->location_truncation(ret, x.s);
    }
    else
    {
        fix->mult->matrix_multiplication(dim1, dim2, dim3, x.data, y.data, ret.data, x.ell, y.ell, ret.ell, true, true,
                                         true, MultMode::None, msb_x, msb_y);
    }
    return ret;
}

FixArray FPMath::zero_sum_modP(size_t row, size_t column, uint64_t prime_mod, int ell, int scale)
{
    FixArray ret(party, row * column, true, ell, scale);
    sci::PRG128 prg;
    prg.random_mod_p(ret.data, row * column, prime_mod);
    size_t i, j;
    int ell_mask = ret.ell_mask();
    vector<FixArray> ret_vec(row, FixArray(ret.party, column, ret.signed_, ret.ell, ret.s));
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < column; j++)
        {
            ret_vec[i].data[j] = ret.data[i * column + j];
        }
    }
    FixArray sum = fix->tree_sum(ret_vec);
    FixArray last = FixArray(party, sum.size, true, ell, scale);
    for (i = 0; i < row; i++)
    {
        last.data[i] = ret.data[(i + 1) * column - 1];
    }
    last = fix->sub(last, sum);
    for (i = 0; i < row; i++)
    {
        ret.data[(i + 1) * column - 1] = last.data[i];
    }
    return ret;
}

FixArray FPMath::gelu_bolt(const FixArray &x)
{
    int N = x.size;
    int ell = x.ell;
    int s = x.s;

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N, 1);

    // Constatns
    // Get y = abs(x)
    // auto start1 = std::chrono::high_resolution_clock::now();
    BoolArray msb_x = fix->MSB(x);
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(end1 - start1);
    // cout << "MSB cost: " << duration1.count() << " sec \n";
    FixArray neg_x = fix->mul(x, -1);
    FixArray abs_x = fix->if_else(msb_x, neg_x, x);
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration2 = std::chrono::duration_cast<std::chrono::duration<double>>(end2 - start1);
    // cout << "ABS cost: " << duration2.count() << " sec \n";

    FixArray cons_1 = fix->input(PUBLIC, N, uint64_t((0.020848611754127593) * (1 << s)), true, ell, s);
    FixArray cons_2 = fix->input(PUBLIC, N, uint64_t((-0.18352506127082727) * (1 << s)), true, ell, s);
    FixArray cons_3 = fix->input(PUBLIC, N, uint64_t((0.5410550166368381) * (1 << s)), true, ell, s);
    FixArray cons_4 = fix->input(PUBLIC, N, uint64_t((-0.03798164612714154) * (1 << s)), true, ell, s);
    FixArray y_right = fix->input(PUBLIC, N, uint64_t((0.001620808531841547) * (1 << s)), true, ell, s);
    // auto start_time = chrono::high_resolution_clock::now();
    FixArray x_square = fix->mul(x, x, ell + s); // x2
    x_square = fix->truncate_reduce(x_square, s);

    FixArray x_cube = fix->mul(x_square, x, ell + s); // x3
    x_cube = fix->truncate_reduce(x_cube, s);

    FixArray x_quartic = fix->mul(x_square, x_square, ell + s); // x4
    x_quartic = fix->truncate_reduce(x_quartic, s);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start_time);
    // cout << "X-X2-X3-X4 cost: " << duration.count() << " sec \n";
    FixArray cons_1_x_quartic = fix->mul(x_quartic, cons_1, ell + s);
    cons_1_x_quartic = fix->truncate_reduce(cons_1_x_quartic, s);

    FixArray cons_2_x_cube = fix->mul(x_cube, cons_2, ell + s);
    cons_2_x_cube = fix->truncate_reduce(cons_2_x_cube, s);

    FixArray cons_3_x_square = fix->mul(x_square, cons_3, ell + s);
    cons_3_x_square = fix->truncate_reduce(cons_3_x_square, s);

    FixArray cons_4_x_single = fix->mul(x, cons_4, ell + s);
    cons_4_x_single = fix->truncate_reduce(cons_4_x_single, s);

    FixArray f_x1_ad_2 = fix->add(cons_1_x_quartic, cons_2_x_cube);
    FixArray f_x_3 = fix->add(f_x1_ad_2, cons_3_x_square); // something_wrong
    FixArray f_x = fix->add(f_x_3, cons_4_x_single);

    // from bolt-gelu_approx_2
    BoolArray lt27 = fix->LT(abs_x, 2.7 * (1 << s));

    FixArray x_plus_y = fix->add(x, abs_x);
    FixArray half_x_plus_y = fix->right_shift(x_plus_y, 1, all_0.data);
    half_x_plus_y.s = s;

    FixArray ret = fix->if_else(lt27, f_x, half_x_plus_y);

    BoolArray msb_ret = bool_op->AND(msb_x, lt27);

    ret = fix->extend(ret, 37, msb_ret.data);

    ret = fix->right_shift(ret, 7, msb_ret.data);
    return ret;
}

vector<FixArray> FPMath::layer_norm_iron(const vector<FixArray> &x, FixArray &w, FixArray &b)
{
    int N = x.size();
    int n = x[0].size;
    int ell = x[0].ell;
    int s = x[0].s;
    bool signed_ = x[0].signed_;

    BoolArray all_0 = bool_op->input(ALICE, N * n, uint8_t(0));
    BoolArray all_1 = bool_op->input(ALICE, N * n, 1);

    FixArray sum = fix->tree_sum(x);

    FixArray dn = fix->input(PUBLIC, sum.size, uint64_t(((1.0 / n) * pow(2, 2 * s))), true, ell, 2 * s);
    FixArray avg = fix->mul(sum, dn, ell + 2 * s, nullptr, all_0.data);
    avg = fix->truncate_reduce(avg, 2 * s);

    FixArray avg_flat(party, N * n, sum.signed_, ell, s);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            avg_flat.data[i * n + j] = avg.data[i];
        }
    }

    FixArray x_flat = concat(x);
    FixArray x_flat_avg = fix->sub(x_flat, avg_flat);
    FixArray x_flat_avg_square = fix->mul(x_flat_avg, x_flat_avg, ell + s);
    x_flat_avg_square = fix->truncate_reduce(x_flat_avg_square, s);

    vector<FixArray> square_group(N);
    for (int i = 0; i < N; i++)
    {
        square_group[i] = FixArray(party, n, signed_, ell, s);
        memcpy(square_group[i].data, &x_flat_avg_square.data[i * n], n * sizeof(uint64_t));
    }

    FixArray square_sum = fix->tree_sum(square_group);
    square_sum = fix->mul(square_sum, dn, ell + 2 * s, all_0.data, all_0.data);
    square_sum = fix->truncate_reduce(square_sum, 2 * s);
    FixArray sigma = sqrt_(square_sum, true);

    FixArray sigma_flat(party, N * n, sum.signed_, ell, s);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sigma_flat.data[i * n + j] = sigma.data[i];
        }
    }

    FixArray x_avg_sigma = fix->mul(x_flat_avg, sigma_flat, ell + s, nullptr, all_0.data);
    x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);

    // Weight and Bias
    x_avg_sigma = fix->mul(x_avg_sigma, w, ell + s);
    x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
    x_avg_sigma = fix->add(x_avg_sigma, b);

    // Hack!
    // x_avg_sigma = fix->extend(x_avg_sigma, 64);

    vector<FixArray> ret(N);
    for (int i = 0; i < N; i++)
    {
        ret[i] = FixArray(party, n, signed_, ell, s);
        memcpy(ret[i].data, &x_avg_sigma.data[i * n], n * sizeof(uint64_t));
    }
    return ret;
}

std::tuple<vector<FixArray>, FixArray> FPMath::softmax_bolt(const vector<FixArray> &x)
{
    // std::cout << "Entering softmax fix" << std::endl;
    int N = x.size();
    int n = x[0].size;
    int ell = x[0].ell;
    int s = x[0].s;

    // for (int i = 0; i < N; i++){
    //   print_fix(x[i]);
    // }

    bool signed_ = x[0].signed_;
    // assert(m_bits > 0);
    for (int i = 1; i < N; i++)
    {
        assert(x[i].party != PUBLIC);
        assert(x[i].ell == ell);
        assert(x[i].s == s);
        assert(x[i].size == n);
    }
    FixArray x_max = fix->max(x);
    // x_max = fix->add(x_max, 1);
    FixArray x_max_flat(party, N * n, signed_, ell, s);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x_max_flat.data[i * n + j] = x_max.data[i];
        }
    }

    // FixArray x_max_flat = fix->input(PUBLIC, N*n, 10<<s, signed_, ell, s);
    // print_fix(x_max_flat);
    // assert(0);

    FixArray x_flat = concat(x);
    FixArray shifted_x_flat = fix->sub(x_flat, x_max_flat);

    FixArray e_x_flat;
    FixArray l_short;

    tie(e_x_flat, l_short) = exp4(shifted_x_flat);
    // FixArray e_x_flat = shifted_x_flat;

    int exp_ell = 19;
    e_x_flat = fix->reduce(e_x_flat, exp_ell);

    vector<FixArray> e_x_tr(n);
    for (int i = 0; i < n; i++)
    {
        e_x_tr[i] = FixArray(party, N, signed_, exp_ell, s);
        for (int j = 0; j < N; j++)
        {
            e_x_tr[i].data[j] = e_x_flat.data[j * n + i];
        }
    }
    FixArray sum_e_x;
    {
        vector<FixArray> tmp = e_x_tr;
        int num_adds_old = n;
        int num_adds_curr = n / 2;
        while (num_adds_old > 1)
        {
            int odd_num_adds = num_adds_old & 1;
            vector<FixArray> lhs(num_adds_curr);
            vector<FixArray> rhs(num_adds_curr);
            for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2)
            {
                lhs[j / 2] = tmp[j];
                rhs[j / 2] = tmp[j + 1];
            }
            FixArray lhs_concat = concat(lhs);
            FixArray rhs_concat = concat(rhs);
            lhs_concat = fix->add(lhs_concat, rhs_concat);
            for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2)
            {
                tmp[odd_num_adds + (j / 2)] = lhs_concat.subset((j / 2) * N, (j / 2) * N + N);
            }
            num_adds_old = num_adds_curr + odd_num_adds;
            num_adds_curr = num_adds_old / 2;
        }
        sum_e_x = tmp[0];
    }

    sum_e_x.signed_ = false;
    FixArray ret_flat = fix->div_batch(e_x_flat, sum_e_x, n, exp_ell, s);

    BoolArray all_0 = bool_op->input(ALICE, N, uint8_t(0));
    ret_flat = fix->extend(ret_flat, ell);

    vector<FixArray> ret(N);
    for (int i = 0; i < N; i++)
    {
        ret[i] = FixArray(party, n, signed_, ell, s);
        memcpy(ret[i].data, ret_flat.data + i * n, n * sizeof(uint64_t));
    }
    return make_tuple(ret, l_short);
}

// double gelu_y = 0.5*dbl_x*(1+tanh(sqrt(2/M_PI)*(dbl_x+0.044715*dbl_x*dbl_x*dbl_x)));
FixArray FPMath::gelu_iron(const FixArray &x)
{
    int N = x.size;
    int ell = x.ell;
    int s = x.s;

    // Constants

    FixArray cons_half = fix->input(PUBLIC, x.size, uint64_t(0.5 * pow(2, s)), false, ell, s);
    FixArray cons_less_half = fix->input(PUBLIC, x.size, uint64_t(0.044715 * pow(2, s)), false, ell, s);
    FixArray cons_some_pi = fix->input(PUBLIC, x.size, uint64_t((std::sqrt(2 / M_PI)) * (1ULL << s)), false, ell, s);

    FixArray x_square = fix->mul(x, x, ell + s);
    x_square = fix->truncate_reduce(x_square, s);

    FixArray x_cube = fix->mul(x_square, x, ell + s);
    x_cube = fix->truncate_reduce(x_cube, s);

    FixArray x_cube_less_half = fix->mul(x_cube, cons_less_half, ell + s);
    x_cube_less_half = fix->truncate_reduce(x_cube_less_half, s);
    x_cube_less_half = fix->add(x_cube_less_half, x);

    FixArray x_cube_pi = fix->mul(x_cube_less_half, cons_some_pi, ell + s);
    x_cube_pi = fix->truncate_reduce(x_cube_pi, s);

    FixArray post_tanh = fix->tanh(x_cube_pi, ell, s);
    post_tanh = fix->add(post_tanh, 1 << s);

    FixArray half_x = fix->mul(x, cons_half, ell + s);
    half_x = fix->truncate_reduce(half_x, s);

    FixArray ret = fix->mul(half_x, post_tanh, ell + s);
    ret = fix->truncate_reduce(ret, s);

    return ret;
}

vector<FixArray> FPMath::softmax_iron(const vector<FixArray> &x)
{
    int N = x.size();
    // for (int i = 0; i < N; i++){
    //   print_fix(x[i]);
    // }
    int n = x[0].size;
    int ell = x[0].ell;
    int s = x[0].s;
    bool signed_ = x[0].signed_;
    // assert(m_bits > 0);
    for (int i = 1; i < N; i++)
    {
        assert(x[i].party != PUBLIC);
        assert(x[i].ell == ell);
        assert(x[i].s == s);
        assert(x[i].size == n);
    }
    FixArray x_max = fix->max_iron(x);
    // print_fix(x_max);
    FixArray x_max_flat(party, N * n, signed_, ell, s);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x_max_flat.data[i * n + j] = x_max.data[i];
        }
    }

    FixArray x_flat = concat(x);
    FixArray shifted_x_flat = fix->sub(x_flat, x_max_flat);

    // FixArray e_x_flat = fix->exp(shifted_x_flat, ell, s);
    FixArray e_x_flat = lookup_table_exp(shifted_x_flat);

    vector<FixArray> e_x_tr(n);
    for (int i = 0; i < n; i++)
    {
        e_x_tr[i] = FixArray(party, N, signed_, ell, s);
        for (int j = 0; j < N; j++)
        {
            e_x_tr[i].data[j] = e_x_flat.data[j * n + i];
        }
    }
    FixArray sum_e_x;
    {
        vector<FixArray> tmp = e_x_tr;
        int num_adds_old = n;
        int num_adds_curr = n / 2;
        while (num_adds_old > 1)
        {
            int odd_num_adds = num_adds_old & 1;
            vector<FixArray> lhs(num_adds_curr);
            vector<FixArray> rhs(num_adds_curr);
            for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2)
            {
                lhs[j / 2] = tmp[j];
                rhs[j / 2] = tmp[j + 1];
            }
            FixArray lhs_concat = concat(lhs);
            FixArray rhs_concat = concat(rhs);
            lhs_concat = fix->add(lhs_concat, rhs_concat);
            for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2)
            {
                tmp[odd_num_adds + (j / 2)] = lhs_concat.subset((j / 2) * N, (j / 2) * N + N);
            }
            num_adds_old = num_adds_curr + odd_num_adds;
            num_adds_curr = num_adds_old / 2;
        }
        sum_e_x = tmp[0];
    }
    FixArray sum_e_x_replicated(party, N * n, signed_, ell, s);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum_e_x_replicated.data[i * n + j] = sum_e_x.data[i];
        }
    }
    sum_e_x_replicated.signed_ = false;
    FixArray ret_flat = fix->div(e_x_flat, sum_e_x_replicated, ell, s);

    // sum_e_x.signed_ = false;
    // FixArray ret_flat = fix->div_batch(e_x_flat, sum_e_x, n ,ell, s);

    // FixArray ret_flat = x_max_flat;
    vector<FixArray> ret(N);
    for (int i = 0; i < N; i++)
    {
        ret[i] = FixArray(party, n, signed_, ell, s);
        memcpy(ret[i].data, ret_flat.data + i * n, n * sizeof(uint64_t));
    }
    // for (int i = 0; i < N; i++){
    //   print_fix(ret[i]);
    // }
    return ret;
}