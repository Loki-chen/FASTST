#include "bert.h"

int main(int argc, const char **argv) {
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
            iopack[i] = new IOPack(party_, 64789 + i, ip);
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

        auto start = get_timestamp();
        Bert *bert = new Bert(party);
        auto end = get_timestamp() - start;
        std::cout << "time of load data: " << end << "\n";

        bool ok = true;
        if (party_ == BOB) {
            iopack[0]->io->send_data(&ok, sizeof(bool));
        } else {
            iopack[0]->io->recv_data(&ok, sizeof(bool));
        }

        std::cout << "time of forward: " << bert->encoders[0]->forward(input, output, fpmath, conv) << "\n";
        // std::cout << "time cost:" << time.count() / 1000000 << "\n";
// 144 986 749 1026
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