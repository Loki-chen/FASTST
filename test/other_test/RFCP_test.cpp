#include <utils.h>
#define N_THREADS 12

int dim[5][3] = {
    { 32,  32, 64},
    {128,  64, 64},
    {128, 256, 64},
    {256, 256, 64},
    {128, 768, 64}
};

int main(int argc, const char **argv) {
    if (argc > 2) {
        int party_ = argv[1][0] - '0', d = argv[2][0] - '0';
        assert(party_ == sci::ALICE || party_ == sci::BOB);
        if (party_ == sci::ALICE) std::cout << "Party: ALICE\n";
        else std::cout << "Party: BOB\n";
        assert(0 <= d && d < 5);
        int dim1 = dim[d][0], dim2 = dim[d][1], dim3 = dim[d][2];
        std::cout << "dim1 = " << dim1 << ", dim2 = " << dim2 << ", dim3 = " << dim3 << "\n";
        BFVParm *parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        BFVKey *party = new BFVKey(party_, parm);
        string ip = "127.0.0.1";
        if (argc > 3) {
            ip = argv[3];
        }

        // sci::IOPack *iopack = new sci::IOPack(party_, 56789, ip);
        sci::IOPack *iopack[N_THREADS];
        for (int i = 0; i < N_THREADS; i++) {
            iopack[i] = new sci::IOPack(party_, 56789 + i, ip);
        }
        bfv_matrix A(dim1 * dim2), B(dim2 * dim3);
        random_bfv_mat(A);
        random_bfv_mat(B);
        size_t start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            start += iopack[i]->get_comm();
        }
        if (party_ == sci::ALICE) {
            auto Ae = RFCP_bfv_encodeA(A, party, dim1, dim2, dim3);
            INIT_TIMER
            START_TIMER

            std::vector<std::thread> send_threads(N_THREADS);
            int split = dim2 / N_THREADS;
            if (split * N_THREADS > dim2) {
                split += 1;
            }
            for (int t = 0; t < N_THREADS; t++) {
                int num_ops = split;
                if (t == N_THREADS - 1) {
                    num_ops = dim2 - split * (N_THREADS - 1);
                }
                send_threads[t] = std::thread([t, num_ops, split, iopack, Ae]() {
                    for (int i = 0; i < num_ops; i++) {
                        BFVLongCiphertext::send(iopack[t]->io, Ae + t * split + i);
                    }
                });
            }
            for (int t = 0; t < N_THREADS; t++) {
                send_threads[t].join();
            }
            STOP_TIMER("RFCP ALICE")
            BFVLongCiphertext C_sec;
            BFVLongCiphertext::recv(iopack[0]->io, &C_sec, parm->context);
            auto C_plain = C_sec.decrypt(party);
            auto C = C_plain.decode_uint(parm);

            delete[] Ae;
        } else {
            BFVLongCiphertext *Ae = new BFVLongCiphertext[dim2];

            std::vector<std::thread> recv_threads(N_THREADS);
            int split = dim2 / N_THREADS;
            if (split * N_THREADS > dim2) {
                split += 1;
            }
            for (int t = 0; t < N_THREADS; t++) {
                int num_ops = split;
                if (t == N_THREADS - 1) {
                    num_ops = dim2 - split * (N_THREADS - 1);
                }
                recv_threads[t] = std::thread([t, num_ops, split, iopack, Ae, parm]() {
                    for (int i = 0; i < num_ops; i++) {
                        BFVLongCiphertext::recv(iopack[t]->io, Ae + t * split + i, parm->context);
                    }
                });
            }
            for (int t = 0; t < N_THREADS; t++) {
                recv_threads[t].join();
            }
            INIT_TIMER
            START_TIMER
            auto res = RFCP_bfv_matmul(Ae, B, dim1, dim2, dim3, parm);
            STOP_TIMER("RFCP")
            delete[] Ae;
            bfv_matrix C(dim1 * dim3);
            random_bfv_mat(C);
            BFVLongPlaintext C_plain(parm, C);
            res.sub_plain_inplace(C_plain, parm->evaluator);
            BFVLongCiphertext::send(iopack[0]->io, &res);
        }
        size_t end = 0;
        for (int i = 0; i < N_THREADS; i++) {
            end += iopack[i]->get_comm();
            delete iopack[i];
        }
        std::cout << "comm: " << end - start << "\n";
        delete party;
        delete parm;
    } else {
        std::cout << "No party input\n";
    }
}