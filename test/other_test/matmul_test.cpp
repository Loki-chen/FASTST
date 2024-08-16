#include <protocols.h>

int dim[4][3] = {
	{64, 64, 128},
	{64, 128, 768},
	{128, 768, 768},
	{128, 768, 3072}
};

int main(int argc, const char** argv) {
	if (argc > 2) {
		int party_ = argv[1][0] - '0';
		assert(party_ == sci::ALICE || party_ == sci::BOB);
		int dim_n = argv[2][0] - '0';
		assert(dim_n > 0 && dim_n < 4);
		int dim1 = dim[dim_n][0], dim2 = dim[dim_n][1], dim3 = dim[dim_n][2];
		if (party_ == sci::ALICE) {
			std::cout << "Party: ALICE\n";
		} else if (party_ == sci::BOB) {
			std::cout << "Party: BOB\n";
		}
		std::cout << "dim1 = " << dim1 << ", dim2 = " << dim2 << ", dim3 = " << dim3 << "\n";
		string ip = "127.0.0.1";
		if (argc > 3) {
			ip = argv[3];
        }
        BFVParm *parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
	    BFVKey *party = new BFVKey(party_, parm);
        sci::IOPack *iopack = new sci::IOPack(party_, 56789, ip);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        sci::NetIO *io = iopack->io;
        Conversion *conv = new Conversion();
        FPMath *fpmath = new FPMath(party_, iopack, otpack);
        FPMath *fpmath_public = new FPMath(sci::PUBLIC, iopack, otpack);

        FixOp *fix_party = new FixOp(party_, iopack, otpack);
        FixOp *fix_public = new FixOp(sci::PUBLIC, iopack, otpack);
        bfv_matrix X(dim1 * dim2), W(dim2 * dim3);
        random_bfv_mat(X);
        random_bfv_mat(W);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0, 1);
        INIT_TIMER
        START_TIMER
        if (party_ == sci::ALICE) {
            double ra = dist(gen);
            uint64_t fix_ra = sci::neg_mod(
                static_cast<int64_t>(ra * (1ULL << (DEFAULT_SCALE))),
                (1ULL << DEFAULT_ELL));
            FixArray fix_xa =
                fpmath->fix->input(sci::ALICE, X.size(), X.data(), true,
                                   DEFAULT_ELL, DEFAULT_SCALE);
            FixArray fix_w =
                fpmath->fix->input(sci::ALICE, W.size(), W.data(), true,
                                   DEFAULT_ELL, DEFAULT_SCALE);

            FixArray fix_ra_xa = fpmath->fix->mul(fix_xa, fix_ra, DEFAULT_ELL),
                     fix_ra_w = fpmath->fix->mul(fix_w, fix_ra, DEFAULT_ELL);
            fix_ra_xa =
                fpmath->fix->location_truncation(fix_ra_xa, DEFAULT_SCALE);
            fix_ra_w =
                fpmath->fix->location_truncation(fix_ra_w, DEFAULT_SCALE);

            fix_ra_xa.party = sci::PUBLIC;
            fix_ra_w.party = sci::PUBLIC;

            FixArray ra_xa_Wa = fpmath->dot(fix_xa, fix_ra_w, dim1, dim2, dim3, DEFAULT_ELL);
            uint64_t ell_mask_ = ra_xa_Wa.ell_mask();
            for (size_t i = 0; i < ra_xa_Wa.size; i++) {
                ra_xa_Wa.data[i] &= ell_mask_;
            }
            BFVLongPlaintext ra_xa_wa_plain(parm, ra_xa_Wa.data, ra_xa_Wa.size);
            BFVLongCiphertext ra_xa_wa_secret_a(party, ra_xa_wa_plain);
            BFVLongCiphertext ra_secret_a(parm, fix_ra, party);
            ra_xa_Wa.party = sci::PUBLIC;
            // fpmath->fix->send_fix_array(ra_xa_Wa);
            BFVLongCiphertext::send(io, &ra_xa_wa_secret_a);
            fpmath->fix->send_fix_array(fix_ra_xa);
            fpmath->fix->send_fix_array(fix_ra_w);
            BFVLongCiphertext::send(io, &ra_secret_a);
        } else {
            // FixArray ra_xa_Wa(sci::PUBLIC, dim1 * dim3, true, DEFAULT_ELL,
            //                   DEFAULT_SCALE),
                fix_ra_xa(sci::PUBLIC, dim1 * dim2, true, DEFAULT_ELL,
                          DEFAULT_SCALE),
                fix_ra_wa(sci::PUBLIC, dim2 * dim3, true, DEFAULT_ELL,
                          DEFAULT_SCALE);
            BFVLongCiphertext ra_xa_wa_secret_a, ra_secret_a;
            BFVLongCiphertext::recv(io, &ra_xa_wa_secret_a, party->parm->context);
            // fpmath->fix->recv_fix_array(ra_xa_Wa);
            fpmath->fix->recv_fix_array(fix_ra_xa);
            fpmath->fix->recv_fix_array(fix_ra_wa);
            BFVLongCiphertext::recv(io, &ra_secret_a, party->parm->context);

            auto cal_raI = [&fpmath, &conv, &party, &fix_ra_xa, &ra_secret_a](
                               FixArray &fix_input, BFVLongCiphertext &ra_xa_WIa_secret_a,
                               const bfv_matrix &WIb, FixArray &ra_WIa) {
                FixArray fix_wib =
                    fpmath->fix->input(sci::BOB, WIb.size(), WIb.data(), true,
                                       DEFAULT_ELL, DEFAULT_SCALE);
                FixArray xb_wib = fpmath->dot(fix_input, fix_wib, dim1, dim2, dim3, DEFAULT_ELL);
                uint64_t *prime_xb_wib = new uint64_t[dim1 * dim3];
                conv->Ring_to_Prime(xb_wib.data, prime_xb_wib, xb_wib.size,
                                    DEFAULT_ELL, party->parm->plain_mod);
                BFVLongPlaintext xb_WIb_plain(party->parm, prime_xb_wib,
                                              xb_wib.size);
                BFVLongCiphertext raI_secret_a = ra_secret_a.multiply_plain(
                    xb_WIb_plain, party->parm->evaluator);
                uint64_t *temp_raI = new uint64_t[dim1 * dim3];
                FixArray temp_raI1 = fpmath->dot(fix_ra_xa, fix_wib, dim1, dim2, dim3, DEFAULT_ELL);
                FixArray temp_raI2 = fpmath->dot(fix_input, ra_WIa, dim1, dim2, dim3, DEFAULT_ELL);
                uint64_t ell_mask_ = temp_raI1.ell_mask();
                for (size_t i = 0; i < dim1; i++) {
                    for (size_t j = 0; j < dim3; j++) {
                        temp_raI[i * dim3 + j] = temp_raI1.data[i * dim3 + j] + temp_raI2.data[i * dim3 + j];
                        temp_raI[i * dim3 + j] &= ell_mask_;
                    }
                }
                conv->Ring_to_Prime(temp_raI, temp_raI, dim1 * dim3,
                                    DEFAULT_ELL, party->parm->plain_mod);
                BFVLongPlaintext temp_raI_plain(party->parm, temp_raI,
                                                dim1 * dim3);
                raI_secret_a.add_plain_inplace(temp_raI_plain,
                                               party->parm->evaluator);
                raI_secret_a.add_inplace(ra_xa_wa_secret_a, party->parm->evaluator);
                delete[] temp_raI;
                delete[] prime_xb_wib;
                return raI_secret_a;
            };
            FixArray fix_x =
                fpmath->fix->input(sci::BOB, X.size(), X.data(), true,
                                   DEFAULT_ELL, DEFAULT_SCALE);
            BFVLongCiphertext raXW_sec_a =
                cal_raI(fix_x, ra_xa_wa_secret_a, W, fix_ra_wa);
        }
        STOP_TIMER("matmul");
        size_t comm = iopack->get_comm();
        size_t rounds = iopack->get_rounds();
        printf("data size of communication: %ld B\n", comm);
        std::cout << "rounds of communication: " << rounds << "\n";

        delete fix_public;
        delete fix_party;
        delete otpack;
        delete iopack;
        delete party;
        delete parm;
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}
