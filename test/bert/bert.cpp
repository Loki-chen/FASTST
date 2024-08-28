#include "bert.h"

void Encoder::forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv) {
    output = vector<uint64_t>(batch_size * d_module);
}

void Bert::forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv) {
    vector<uint64_t> upper_out;
    encoders[0]->forward(input, upper_out, fpmath, conv);
    for (int i = 0; i < n_layer; i++) {
        encoders[i]->forward(upper_out, output, fpmath, conv);
        upper_out = output;
    }
}