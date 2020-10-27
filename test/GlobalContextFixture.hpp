#pragma once

#include "gpi/Context.hpp"

namespace tarantella
{
  class GlobalContext
  {
    public:

      GlobalContext()
      {
        instance() = this;
      }
      static GlobalContext*& instance()
      {
        static GlobalContext* s_inst = 0;
        return s_inst;
      }

      tarantella::GPI::Context gpi_cont;
  };
}
