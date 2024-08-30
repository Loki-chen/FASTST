#include <utils.h>

int main(int argc, const char **argv) {
    int party_ = argv[1][0] - '0';
    BFVParm *parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
    BFVKey *party = new BFVKey(party_, parm);
    vector<uint64_t> data = {1, 2, 3};
    BFVLongPlaintext data_plain(parm, data);
    IOPack *iopack = new IOPack(party_, 56789);
    OTPack *otpack = new OTPack(iopack, party_);
    NetIO *io = iopack->io;
    if (party_ == sci::ALICE) {
        BFVLongCiphertext data_sec(data_plain, party);
        BFVLongCiphertext::send(io, &data_sec);
        BFVLongCiphertext ret_sec;
        BFVLongCiphertext::recv(io, &ret_sec, parm->context);
        auto ret_plain = ret_sec.decrypt(party);
        auto ret = ret_plain.decode_uint(parm);
        std::cout << ret[0] << " " << ret[1] << " " << ret[2] << "\n";
    } else {
        BFVLongCiphertext data_sec_a;
        BFVLongCiphertext::recv(io, &data_sec_a, parm->context);
        data_sec_a.add_plain_inplace(data_plain, parm->evaluator);
        BFVLongCiphertext::send(io, &data_sec_a);
    }
}