#include "mailBoxLocal.h"

namespace tarantella
{
  namespace collectives
  {
    mailBoxLocal::mailBoxLocal()
    : status(0),
      target(0) {}

    bool mailBoxLocal::gotNotification() {
      unsigned long statusOld = status;
      return (statusOld < target) && status.compare_exchange_strong(statusOld, statusOld + 1);
    }

    void mailBoxLocal::notify() {
      ++target;
    }
  }
}
