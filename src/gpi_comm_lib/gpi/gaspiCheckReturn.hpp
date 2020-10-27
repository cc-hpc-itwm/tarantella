#pragma once

#include <GASPI.h>

#include <string>
#include <cstdlib>
#include <stdexcept>

namespace tarantella
{
  namespace GPI
  {
    void gaspiCheckReturn(const gaspi_return_t err,
                          const std::string prefix);
  }
}