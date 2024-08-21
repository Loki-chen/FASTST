#include <utils.h>

using namespace sci;

int main(int argc, const char **argv) {
    if (argc > 1) {
        int party_ = argv[1][0] - '0';
        assert(party_ == ALICE || party_ == BOB);
        party_ == ALICE ? std::cout << "Party: ALICE\n" : std::cout << "Party: BOB\n";
        string ip = "127.0.0.1";
        if (argc > 2) {
            ip = argv[2];
        }
        IOPack *iopack = new IOPack(party_, 56789, ip);
        OTPack *otpack = new OTPack(iopack, party_);
        NetIO *io = iopack->io;
        FPMath *fpmath = new FPMath(party_, iopack, otpack);
        Conversion *conv = new Conversion();
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);
    } else {
        std::cout << "No party input\n";
    }
    return 0;
}