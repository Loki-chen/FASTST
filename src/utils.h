#ifndef FAST_UTILS_H__
#define FAST_UTILS_H__
#pragma once
#include "utils/he-tools.h"
#include "utils/he-bfv.h"
#include "utils/mat-tools.h"
#include "utils/conversion.h"

inline string replace(string str, string substr1, string substr2) {
    size_t index = str.find(substr1);
    str.replace(index, substr1.length(), substr2);
    return str;
}
#endif