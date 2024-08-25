#include <utils.h>

int dim[5][3] = {
    {32, 32, 64},
    {128, 64, 64},
    {128, 256, 64},
    {128, 256, 256},
    {128, 768, 64}
};

int main(int argc, const char** argv) {
    if (argc > 2) {
		int party_ = argv[1][0] - '0', d = argv[2][0] - '0';
		assert(party_ == sci::ALICE || party_ == sci::BOB);
		if (party_ == sci::ALICE) std::cout << "Party: ALICE\n"; else std::cout << "Party: BOB\n";
        assert(0 <= d && d < 5);
        int dim1 = dim[d][0], dim2 = dim[d][1], dim3 = dim[d][2];
        std::cout << "dim1 = " << dim1 << ", dim2 = " << dim2 << ", dim3 = " << dim3 << "\n";
        BFVParm *parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
	    BFVKey *party = new BFVKey(party_, parm);
        string ip = "127.0.0.1";
        if (argc > 3) {
            ip = argv[3];
        }
        sci::IOPack *iopack = new sci::IOPack(party_, 56789, ip);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        sci::NetIO *io = iopack->io;
        bfv_matrix A(dim1 * dim2), B(dim2 * dim3);
        random_bfv_mat(A);
        random_bfv_mat(B);
        size_t start = iopack->get_comm();
        if (party_ == sci::ALICE) {
            auto Ae = RFCP_bfv_encodeA(A, party, dim1, dim2, dim3);
            for (int i = 0; i < dim2; i++) {
                BFVLongCiphertext::send(io, Ae + i);
            }
            delete[] Ae;
            BFVLongCiphertext C_sec;
            INIT_TIMER
            START_TIMER
            BFVLongCiphertext::recv(io, &C_sec, parm->context);
            auto C_plain = C_sec.decrypt(party);
            auto C = C_plain.decode_uint(parm);
            STOP_TIMER("RFCP")
        } else {
            BFVLongCiphertext* Ae = new BFVLongCiphertext[dim2];
            for (int i = 0; i < dim2; i++) {
                BFVLongCiphertext::recv(io, Ae + i, parm->context);
            }
            INIT_TIMER
            START_TIMER
            auto res = RFCP_bfv_matmul(Ae, B, dim1, dim2, dim3, parm);
            delete[] Ae;
            bfv_matrix C(dim1 * dim3);
            random_bfv_mat(C);
            BFVLongPlaintext C_plain(parm, C);
            res.sub_plain_inplace(C_plain, parm->evaluator);
            BFVLongCiphertext::send(io, &res);
            STOP_TIMER("RFCP")
        }
        std::cout << "comm: " << iopack->get_comm() - start << "\n";
    } else { std::cout << "No party input\n"; }
}
// ./BOLT-softmax r=1 ip=172.20.0.2;./BOLT-layer_norm r=1 ip=172.20.0.2;./IRON-softmax r=1 ip=172.20.0.2;./IRON-layer_norm r=1 ip=172.20.0.2;./IRON-gelu r=1 ip=172.20.0.2
// ./BOLT-softmax r=2 ip=172.20.0.3;./BOLT-layer_norm r=2 ip=172.20.0.3;./IRON-softmax r=2 ip=172.20.0.3;./IRON-layer_norm r=2 ip=172.20.0.3;./IRON-gelu r=2 ip=172.20.0.3