#pragma once

#include "mailBox.h"

#include <GASPI.h>

namespace tarantella
{
  namespace collectives
  {
    class mailBoxGaspi : public mailBox 
    {
      public:
        mailBoxGaspi(const gaspi_segment_id_t segmentID_,
                    const gaspi_notification_id_t mailID_);
        bool gotNotification() override;
        gaspi_segment_id_t getSegmentID() const;
        gaspi_notification_id_t getMailID() const;

      private:

        const gaspi_segment_id_t segmentID;
        const gaspi_notification_id_t mailID;
    };
  }
}
