#pragma once

#include "mailBox.h"
#include <atomic>

namespace tarantella
{
  namespace collectives
  {
    class mailBoxLocal : public mailBox 
    {
      public:
        mailBoxLocal();
        bool gotNotification() override;
        void notify();

      private:
        std::atomic<unsigned long> status;
        std::atomic<unsigned long> target;
    };
  }
}
