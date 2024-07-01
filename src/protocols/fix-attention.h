
#ifndef FAST_FIXED_ATTENTION_H__
#define FAST_FIXED_ATTENTION_H__
#pragma once
#include "fixed-protocol.h"
class Fixed_Multi_Head_Attention;

class Fixed_Attention : public FixedProtocol
{
    int head;

public:
    friend Fixed_Multi_Head_Attention;
    Fixed_Attention();
    ~Fixed_Attention();
    bfv_matrix forward(const bfv_matrix &input) const;
}
#endif // FAST_FIXED_ATTENTION_H__