#ifndef FAST_MODEL_H__
#define FAST_MODEL_H__
#include "utils.h"
#define LOG
#define WARNING
#define TRANSFORMER_LOG
// Bert base
// #define batch_size 128 // 影响不大
// #define d_module 768   // 5.5X  9.5x
// #define n_heads 12     // 影响不大
// #define ffn_dim 3072   // 影响不大

// test condition
#define batch_size 128 // 影响不大
#define d_module 768   // 5.5X  9.5x
#define n_heads 12     // 影响不大
#define d_k 64
#define ffn_dim 3072 // 影响不大

// test condition
// #define batch_size  // 影响不大
// #define d_module 5   // 5.5X  9.5x
// #define n_heads 12   // 影响不大
// #define d_k 64
// #define ffn_dim 3072 // 影响不大

#define n_layer 12

#endif