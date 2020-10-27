#pragma once

namespace tarantella
{
  namespace collectives
  {
    class mailBox 
    {
      public:
        virtual bool gotNotification() = 0;
        virtual ~mailBox() = default;
    };
  }
}
