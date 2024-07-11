#include <protocols.h>

int main(int argc, const char **argv) {
    if (argc > 1) {
        int party_ = argv[1][0] - '0';
        assert(party_ == sci::ALICE || party_ == sci::BOB);
        if (party_ == sci::ALICE) {
            std::cout << "Party: ALICE"
                      << "\n";
        } else if (party_ == sci::BOB) {
            std::cout << "Party: BOB"
                      << "\n";
        }
        if (argc > 2) {
            ip = argv[2];
        } else {
            ip = "127.0.0.1";
        }
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);
        sci::IOPack *iopack = new sci::IOPack(party_, 56789, ip);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        sci::NetIO *io = iopack->io;
        Conversion *conv = new Conversion();
        FPMath *fpmath = new FPMath(party_, iopack, otpack);
        FPMath *fpmath_public = new FPMath(sci::PUBLIC, iopack, otpack);

        FixOp *fix_party = new FixOp(party_, iopack, otpack);
        FixOp *fix_public = new FixOp(sci::PUBLIC, iopack, otpack);
        bfv_matrix input(batch_size * d_module);
        random_bfv_mat(input);

        FixedFFN *ffn = new FixedFFN(0, party, bfv_parm, io, fpmath, fpmath_public, conv);
        BFVLongCiphertext ln_secret;
        if (party_ == sci::ALICE) {
            bfv_matrix ln(batch_size * d_module);
            random_bfv_mat(ln);

            BFVLongPlaintext ln_plain(bfv_parm, ln);
            BFVLongCiphertext ln_s_a(ln_plain, party);
            BFVLongCiphertext::send(iopack->io, &ln_s_a);

        } else if (party_ == sci::BOB) {
            BFVLongCiphertext::recv(iopack->io, &ln_secret, bfv_parm->context);
        }
        printf("batch size:       %d\nd_module:         %d\nFFN_dim:          %d\n", batch_size, d_module, ffn_dim);
        BFVLongCiphertext result = ffn->forward(ln_secret);
        size_t comm = iopack->get_comm();
        size_t rounds = iopack->get_rounds();
        if (comm < 1024) {
            printf("data size of communication: %ld B\n", comm);
        } else if (comm < 1024 * 1024) {
            printf("data size of communication: %.2lf KB\n", comm / 1024.);
        } else if (comm < 1024 * 1024 * 1024) {
            printf("data size of communication: %.2lf MB\n", comm / (1024. * 1024.));
        } else {
            printf("data size of communication: %.2lf MB\n", comm / (1024. * 1024. * 1024.));
        }
        std::cout << "rounds of communication: " << rounds << "\n";

        delete ffn;
        delete fix_public;
        delete fix_party;
        delete otpack;
        delete iopack;
        delete party;
        delete bfv_parm;
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}