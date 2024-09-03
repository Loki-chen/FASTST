#include "protocols/fixed-protocol.h"
#include <utils.h>
#define N_THREADS 12

using namespace sci;

int main(int argc, const char **argv) {
    timestamp p_r_time = 0, t_start;
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
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, bfv_parm);

        int dim1 = 128, dim2 = 128;
        vector<uint64_t> input1(128 * 128), input2(128), input3(128 * 3072), input4(128);
        random_ell_mat(input1, DEFAULT_ELL);
        random_ell_mat(input2, DEFAULT_ELL);
        random_ell_mat(input3, DEFAULT_ELL);
        random_ell_mat(input4, DEFAULT_ELL);
        size_t start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            start += iopack[i]->get_comm();
        }
        for (int head = 0; head < 12; head++) {
            std::cout << "Prime-to-Ring " << head << " start\n";
            t_start = TIME_STAMP;
            conv->Prime_to_Ring(party_, N_THREADS, input1.data(), input1.data(), input1.size(), DEFAULT_ELL,
                                bfv_parm->plain_mod, DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
            conv->Prime_to_Ring(party_, N_THREADS, input2.data(), input2.data(), input2.size(), DEFAULT_ELL,
                                bfv_parm->plain_mod, DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
            conv->Prime_to_Ring(party_, N_THREADS, input3.data(), input3.data(), input3.size(), DEFAULT_ELL,
                                bfv_parm->plain_mod, DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
            conv->Prime_to_Ring(party_, N_THREADS, input4.data(), input4.data(), input4.size(), DEFAULT_ELL,
                                bfv_parm->plain_mod, DEFAULT_SCALE, DEFAULT_SCALE, fpmath);
            p_r_time += (TIME_STAMP - t_start);
            std::cout << "Prime-to-Ring " << head << " end\n";
        }
        size_t end = 0;
        for (int i = 0; i < N_THREADS; i++) {
            end += iopack[i]->get_comm();
        }
        std::cout << "time: " << p_r_time << "\n";
        std::cout << "comm: " << end - start << "\n";

        delete party;
        delete bfv_parm;
        delete conv;
        for (int i = 0; i < N_THREADS; i++) {
            delete iopack[i];
            delete otpack[i];
            delete fpmath[i];
        }
    } else {
        std::cout << "No party input\n";
    }
}