#pragma once

#include "tensorflow/core/public/version.h"

using TFLongIntType = typename std::conditional_t<TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >=5,
                                                  int64_t,
                                                  long long int>; // tensorflow::int64 [deprecated in TF 2.7]

