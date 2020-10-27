#include "gaspiCheckReturn.hpp"

#include <string>
#include <cstdlib>
#include <stdexcept>

namespace tarantella
{
  namespace GPI
  {
    void gaspiCheckReturn(const gaspi_return_t err,
                          const std::string prefix)
    {
      if (err != GASPI_SUCCESS)
      {
        gaspi_string_t raw;
        gaspi_print_error(err, &raw);
        std::string message = prefix + std::string(raw);
        free(raw);
        throw std::runtime_error(message);
      }
    }
  }
}
