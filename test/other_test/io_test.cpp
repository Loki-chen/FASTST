#include <ezpc_scilib/ezpc_utils.h>
#include <utils.h>

int main(int argc, const char **argv)
{
    if (argc > 1)
    {
        int party_ = argv[1][0] - '0';
        assert(party_ == sci::ALICE || party_ == sci::BOB);
        if (party_ == sci::ALICE)
        {
            std::cout << "Party: ALICE\n\n";
        }
        else if (party_ == sci::BOB)
        {
            std::cout << "Party: BOB\n\n";
        }
        BFVParm *bfv_parm = new BFVParm(8192, {54, 54, 55, 55}, default_prime_mod.at(29));
        sci::IOPack *iopack = new sci::IOPack(party_, 56789);
        sci::OTPack *otpack = new sci::OTPack(iopack, party_);
        sci::NetIO *io = iopack->io;
        FixOp *fix_public = new FixOp(sci::PUBLIC, iopack, otpack);
        int size = 10000;

        BFVLongCiphertext attn_secret_b;
        std::cout << YELLOW << "size: " << size << RESET << "\n";
        if (party_ == sci::ALICE)
        {
            std::cout << YELLOW << "test FixArray io pack:\n"
                      << RESET << "-------------------------------\n";
            bfv_matrix io_test(size);
            random_bfv_mat(io_test);
            FixArray fix_io_test = fix_public->input(sci::PUBLIC, size, io_test.data(), true, 32, 63);
            int comm_data = io->counter;
            fix_public->send_fix_array(fix_io_test);
            int send = io->counter - comm_data;
            std::cout << "send data:        " << send << "\n";
            int theoretical = 4 * sizeof(int) + sizeof(bool) + sizeof(uint64_t) * size;
            std::cout << "theoretical data: " << theoretical << "\n";
            std::cout << "all comm data:    " << io->counter << "\n";
            send == theoretical ? std::cout << GREEN << "test pass\n"
                                            << RESET
                                : std::cout << RED << "test not pass\n"
                                            << RESET;
            std::cout << "-------------------------------\n";

            std::cout << YELLOW << "test BFVLongCiphertext io pack:\n"
                      << RESET << "-------------------------------\n";
            BFVLongCiphertext test_secret;
            comm_data = io->counter;
            BFVLongCiphertext::recv(io, &test_secret, bfv_parm->context);
            send = io->counter - comm_data;
            std::cout << "send data:        " << send << "\n";
            std::cout << "theoretical data: 0\n";
            std::cout << "all comm data:    " << io->counter << "\n";
            send == 0 ? std::cout << GREEN << "test pass\n"
                                  << RESET
                      : std::cout << RED << "test not pass\n"
                                  << RESET;
            std::cout << "-------------------------------\n";
        }
        else if (party_ == sci::BOB)
        {
            std::cout << YELLOW << "test FixArray io pack:\n"
                      << RESET << "-------------------------------\n";
            FixArray fix_io_test(sci::PUBLIC, size, true, 32, 63);
            int comm_data = io->counter;
            fix_public->recv_fix_array(fix_io_test);
            int send = io->counter - comm_data;
            std::cout << "send data:        " << send << "\n";
            std::cout << "theoretical data: 0\n";
            std::cout << "all comm data:    " << io->counter << "\n";
            send == 0 ? std::cout << GREEN << "test pass\n"
                                  << RESET
                      : std::cout << RED << "test not pass\n"
                                  << RESET;
            std::cout << "-------------------------------\n";

            std::cout << YELLOW << "test BFVLongCiphertext io pack:\n"
                      << RESET << "-------------------------------\n";
            BFVKey *party = new BFVKey(party_, bfv_parm);
            BFVLongPlaintext test_plain(bfv_parm, fix_io_test.data, fix_io_test.size);
            BFVLongCiphertext test_secret(test_plain, party);
            comm_data = io->counter;
            BFVLongCiphertext::send(io, &test_secret);
            send = io->counter - comm_data;
            std::cout << "send data:        " << send << "\n";
            int theoretical = sizeof(size_t) * 2;
            for (auto &ct : test_secret.cipher_data)
            {
                std::stringstream os;
                uint64_t ct_size;
                ct.save(os);
                ct_size = os.tellp();
                string ct_ser = os.str();
                theoretical += sizeof(uint64_t);
                theoretical += ct_ser.size();
            }
            std::cout << "theoretical data: " << theoretical << "\n";
            std::cout << "all comm data:    " << io->counter << "\n";
            send == theoretical ? std::cout << GREEN << "test pass\n"
                                            << RESET
                                : std::cout << RED << "test not pass\n"
                                            << RESET;
            std::cout << "-------------------------------\n";
        }
    }
    else
    {
        std::cout << "No party input\n";
    }
}