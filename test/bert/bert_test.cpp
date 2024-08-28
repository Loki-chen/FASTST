#include "bert.h"

int main(int argc, const char** argv) {
    if (argc > 1) {
        int party_ = argv[1][0] - '0';
        assert(party_ == ALICE || party_ == BOB);
        party_ == ALICE ? std::cout << "Party: ALICE\n" : std::cout << "Party: BOB\n";
        string ip = "127.0.0.1";
        if (argc > 2) {
            ip = argv[2];
        }
        IOPack *iopack[N_THREADS]; // = new IOPack(party_, 56789, ip);
        OTPack *otpack[N_THREADS]; // = new OTPack(iopack, party_);
        FPMath *fpmath[N_THREADS]; // = new FPMath(party_, iopack, otpack);
        for (int i = 0; i < N_THREADS; i++) {
            iopack[i] = new IOPack(party_, 56789 + i, ip);
            otpack[i] = new OTPack(iopack[i], party_);
            fpmath[i] = new FPMath(party_, iopack[i], otpack[i]);
        }

        Conversion *conv = new Conversion();
        BFVParm *parm =
            new BFVParm(8192, {40, 30, 30, 40},
                        default_prime_mod.at(29)); // new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, parm);

        vector<uint64_t> input(batch_size * d_module), output;
        random_ell_mat(input, DEFAULT_ELL);

        INIT_TIMER
        START_TIMER
        Bert *bert = new Bert(party);
        STOP_TIMER("load data")

        START_TIMER
        bert->forward(input, output, fpmath, conv);
        STOP_TIMER("bert")

        delete bert;
        delete party;
        delete parm;
        delete conv;
        for (int i = 0; i < N_THREADS; i++) {
            delete fpmath[i];
            delete otpack[i];
            delete iopack[i];
        }
    } else {
        std::cout << "No party input\n";
    }
}