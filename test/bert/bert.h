#include <model.h>
#include <utils.h>
#define DEFAULT_SCALE 12
#define DEFAULT_ELL 37
#define N_THREADS 12

using namespace sci;

const string base_path = "/data/BOLT/bolt/prune/mrpc/weights_txt/";
const string WQ_path = "bert.encoder.layer.LAYER.attention.self.query.weight.txt",
             WK_path = "bert.encoder.layer.LAYER.attention.self.key.weight.txt",
             WV_path = "bert.encoder.layer.LAYER.attention.self.value.weight.txt",
             bQ_path = "bert.encoder.layer.LAYER.attention.self.query.bias.txt",
             bK_path = "bert.encoder.layer.LAYER.attention.self.key.bias.txt",
             bV_path = "bert.encoder.layer.LAYER.attention.self.value.bias.txt",
             Attn_W_path = "bert.encoder.layer.LAYER.attention.output.dense.weight.txt",
             Attn_b_path = "bert.encoder.layer.LAYER.attention.output.dense.bias.txt",
             gamma1_path = "bert.encoder.layer.LAYER.attention.output.LayerNorm.weight.txt",
             beta1_path = "bert.encoder.layer.LAYER.attention.output.LayerNorm.bias.txt",
             W1_path = "bert.encoder.layer.LAYER.intermediate.dense.weight.txt",
             b1_path = "bert.encoder.layer.LAYER.intermediate.dense.bias.txt",
             W2_path = "bert.encoder.layer.LAYER.output.dense.weight.txt",
             b2_path = "bert.encoder.layer.LAYER.output.dense.bias.txt",
             gamma2_path = "bert.encoder.layer.LAYER.output.LayerNorm.weight.txt",
             beta2_path = "bert.encoder.layer.LAYER.output.LayerNorm.bias.txt";

class Encoder {
public:
    BFVKey *party;
    int layer;
    vector<uint64_t> WQ, WK, WV, bQ, bK, bV, Attn_W, Attn_b, gamma1, beta1, W1, b1, W2, b2, gamma2, beta2;
    Encoder(BFVKey *_party, int _layer) : party(_party), layer(_layer) {
        if (party->party == BOB) {
            string layer_str = std::to_string(layer);
            load_bfv_mat(WQ, replace(base_path + WQ_path, "LAYER", layer_str));
            load_bfv_mat(WK, replace(base_path + WK_path, "LAYER", layer_str));
            load_bfv_mat(WV, replace(base_path + WV_path, "LAYER", layer_str));
            load_bfv_mat(bQ, replace(base_path + bQ_path, "LAYER", layer_str));
            load_bfv_mat(bK, replace(base_path + bK_path, "LAYER", layer_str));
            load_bfv_mat(bV, replace(base_path + bV_path, "LAYER", layer_str));
            load_bfv_mat(Attn_W, replace(base_path + Attn_W_path, "LAYER", layer_str));
            load_bfv_mat(Attn_b, replace(base_path + Attn_b_path, "LAYER", layer_str));
            load_bfv_mat(gamma1, replace(base_path + gamma1_path, "LAYER", layer_str));
            load_bfv_mat(beta1, replace(base_path + beta1_path, "LAYER", layer_str));
            load_bfv_mat(W1, replace(base_path + W1_path, "LAYER", layer_str));
            load_bfv_mat(b1, replace(base_path + b1_path, "LAYER", layer_str));
            load_bfv_mat(W2, replace(base_path + W2_path, "LAYER", layer_str));
            load_bfv_mat(b2, replace(base_path + b2_path, "LAYER", layer_str));
            load_bfv_mat(gamma2, replace(base_path + gamma1_path, "LAYER", layer_str));
            load_bfv_mat(beta2, replace(base_path + beta1_path, "LAYER", layer_str));
        }
    }
    void forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv);
};

class Bert {
public:
    BFVKey *party;
    vector<Encoder *> encoders;
    Bert(BFVKey *_party) : party(_party) {
        encoders.resize(n_layer);
        for (int i = 0; i < n_layer; i++) {
            encoders[i] = new Encoder(party, i);
        }
    }
    ~Bert() {
        for (int i = 0; i < n_layer; i++) {
            delete encoders[i];
        }
    }
    void forward(const vector<uint64_t> &input, vector<uint64_t> &output, FPMath **fpmath, Conversion *conv);
};