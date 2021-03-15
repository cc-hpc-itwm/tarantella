#pragma once

#include <GaspiCxx/Runtime.hpp>

namespace tarantella
{
  struct GlobalContext {
    GlobalContext()
    {
      gaspi::initGaspiCxx();
    }
    ~GlobalContext() = default;
  };
}
