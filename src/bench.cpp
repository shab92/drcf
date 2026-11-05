// Benchmark: DistAEAD.Dec / Enc cost vs message size. Run: ./bench 1 & ./bench 2
#include <chrono>
#include <cstdio>
#include "distaead.h"
using namespace emp;
using bytes = std::vector<uint8_t>;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port(), reps = argc > 2 ? atoi(argv[2]) : 10;
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    bytes key(16, party), nonce(16, 7), ad = {'t', 'o', 'p', 'i', 'c'};
    if (party == ALICE) printf("%-6s %7s %10s %12s %10s\n", "op", "bytes", "AND gates", "comm (KiB)", "ms/op");
    for (size_t len : {8, 16, 32, 64, 128, 256}) {
        bytes share(len, 0x5a);
        for (int op = 0; op < 2; op++) {
            // warm up OT extension buffers
            if (op == 0) distaead::dist_enc(s, key, nonce.data(), ad, share);
            bytes ct(len + 16, 0x33);
            uint64_t a0 = s.num_and(), c0 = io->send_counter + io->recv_counter;
            auto t0 = clk::now();
            for (int r = 0; r < reps; r++) {
                if (op == 0) distaead::dist_enc(s, key, nonce.data(), ad, share);
                else distaead::dist_dec(s, key, nonce.data(), ad, ct);
            }
            io->flush();
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
            if (party == ALICE)
                printf("%-6s %7zu %10llu %12.1f %10.2f\n", op ? "Dec" : "Enc", len, (unsigned long long)((s.num_and() - a0) / reps),
                       (io->send_counter + io->recv_counter - c0) / 1024.0 / reps, ms);
        }
    }
}
