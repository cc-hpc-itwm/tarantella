#include "mailBoxGaspi.h"
#include "gpi/gaspiCheckReturn.hpp"

#include <cassert>

namespace tarantella
{
  namespace collectives
  {
    using tarantella::GPI::gaspiCheckReturn;

    mailBoxGaspi::mailBoxGaspi(const gaspi_segment_id_t segmentID_,
                              const gaspi_notification_id_t mailID_)
    : segmentID(segmentID_),
      mailID(mailID_) {}

    bool mailBoxGaspi::gotNotification() {
      gaspi_notification_id_t event;
      gaspi_return_t err = gaspi_notify_waitsome(segmentID,
                                                mailID,
                                                1,
                                                &event,
                                                GASPI_TEST);
      if (err == GASPI_TIMEOUT)
      {
        return false;
      }
      gaspiCheckReturn(err, "gaspi_notify_waitsome failed with ");

      assert(mailID == event);
      gaspi_notification_t value;
      gaspiCheckReturn(gaspi_notify_reset(segmentID,
                                          event,
                                          &value),
                      "gaspi_notify_reset failed with ");
      return value != 0;
    }

    gaspi_segment_id_t mailBoxGaspi::getSegmentID() const {
      return segmentID;
    }

    gaspi_notification_id_t mailBoxGaspi::getMailID() const {
      return mailID;
    }
  }
}
