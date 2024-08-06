#include "utils.h"
#include "gmp.h"
#include "cmath"

void matrixMultiplication(vector<int64_t> &mat1, vector<int64_t> &mat2, vector<int64_t> &mat3)
{
    for (size_t i = 0; i < mat1.size(); i++)
    {
        mat3[i] = mat1[i] * mat2[i];
    }
}

void random_mat(vector<int64_t> &mat, double min, double max, int scale)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    size_t size = mat.size();
    for (size_t i = 0; i < size; i++)
    {
        mat[i] = static_cast<int64_t>(dist(gen) * pow(2, scale));
    }
}

int main()
{

    /*
    init possess........
    */

    int dim_a = 2, dim_b = 2, dim_c = 2;

    int64_t field = 4294967311,
            scale = 12; //~ 31-bit
    // int64_t field = 2061584302081; ~ 41-bit

    BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(31));
    BFVKey *party = new BFVKey(1, bfv_parm);

    vector<int64_t> x_a(dim_a * dim_b);
    vector<int64_t> w_a(dim_b * dim_c);
    vector<int64_t> x_b(dim_a * dim_b);
    vector<int64_t> w_b(dim_b * dim_c);
    for (size_t i = 0; i < dim_a * dim_b; i++)
    {
        /* code */
    }

    // ALICE
    random_mat(x_a, -1, 1, scale);
    random_mat(w_a, -1, 1, scale);

    sci::OTPack *otpack;
    sci::IOPack *iopack;

    FixFieldOp *fix = new FixFieldOp(sci::PUBLIC, iopack, otpack);

    FixFieldArray prime_xa = fix->input(sci::PUBLIC, dim_a * dim_b, x_a.data(), field, scale);
    FixFieldArray prime_wa = fix->input(sci::PUBLIC, dim_b * dim_c, w_a.data(), field, scale);

    FixFieldArray prime_xawa = fix->mul(prime_xa, prime_wa, 2 * field);
    BFVLongPlaintext enc_xa(bfv_parm, prime_xa.data, dim_a * dim_b), enc_wa(bfv_parm, prime_wa.data, dim_b * dim_c), enc_xawa(bfv_parm, prime_xawa.data, dim_a * dim_c);
    BFVLongCiphertext hexa(enc_xa, party), hewa(enc_wa, party), hexawa(enc_xawa, party);

    // BOB
    random_mat(x_b, -1, 1, scale);
    random_mat(w_b, -1, 1, scale);
    int dim = dim_a * dim_b;
    vector<int64_t> ture_ret1(dim);
    matrixMultiplication(x_b, w_b, ture_ret1);
    vector<int64_t> ture_ret2(dim);
    matrixMultiplication(x_a, w_a, ture_ret2);
    vector<int64_t> ture_ret3(dim);
    matrixMultiplication(x_a, w_b, ture_ret3);
    vector<int64_t> ture_ret4(dim);
    matrixMultiplication(w_a, x_b, ture_ret4);

    vector<int64_t> true_ret(dim_a * dim_b);
    std::cout << "true result: ";
    for (size_t i = 0; i < dim_a * dim_b - 2; i++)
    {
        true_ret[i] = ture_ret1[i] + ture_ret2[i] + ture_ret3[i] + ture_ret4[i];
        std::cout << true_ret[i] / pow(2, 24) << " ";
    }
    std::cout << std::endl;
    FixFieldArray prime_xb = fix->input(sci::PUBLIC, dim_a * dim_b, x_b.data(), field, scale);
    FixFieldArray prime_wb = fix->input(sci::PUBLIC, dim_b * dim_c, w_b.data(), field, scale);

    std::cout << prime_xb.data[0] << " " << prime_xb.data[1] << "\n";
    std::cout << prime_wb.data[0] << " " << prime_wb.data[1] << "\n";
    FixFieldArray prime_xbwb = fix->mul(prime_xb, prime_wb, 2 * field);
    std::cout << prime_xbwb.data[0] << " " << prime_xbwb.data[1] << "\n";
    std::cout << static_cast<int64_t>(prime_xbwb.data[0] / pow(2, 12)) << " " << static_cast<int64_t>(prime_xbwb.data[1] / pow(2, 12)) << "\n";

    BFVLongPlaintext enc_xb(bfv_parm, prime_xb.data, dim_a * dim_b),
        enc_wb(bfv_parm, prime_wb.data, dim_b * dim_c), enc_xbwb(bfv_parm, prime_xbwb.data, dim_a * dim_c);
    BFVLongCiphertext xawb = hexa.multiply_plain(enc_wb, bfv_parm->evaluator);
    BFVLongCiphertext waxb = hewa.multiply_plain(enc_xb, bfv_parm->evaluator);

    BFVLongPlaintext test_waxb = waxb.decrypt(party);
    vector<int64_t> ret = plain_ret.decode_int(bfv_parm);

    BFVLongCiphertext ret1 = xawb.add(hexawa, bfv_parm->evaluator); // somtthings wrong here with different dim

    BFVLongCiphertext ret2 = ret1.add(waxb, bfv_parm->evaluator);

    BFVLongCiphertext enc_ret = ret2.add_plain(enc_xbwb, bfv_parm->evaluator);

    BFVLongPlaintext plain_ret = enc_ret.decrypt(party);

    vector<int64_t> ret = plain_ret.decode_int(bfv_parm);

    std::cout << "result: ";
    for (size_t i = 0; i < ret.size() - 2; i++)
    {
        std::cout << ret[i] / pow(2, 12) << " ";
    }
    std::cout << std::endl;
    return 0;
}
