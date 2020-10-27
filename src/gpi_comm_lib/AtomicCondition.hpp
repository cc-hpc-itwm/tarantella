#pragma once

#include <condition_variable>
#include <mutex>

class AtomicCondition
{
  public:
    void notify()
    {
      {
        std::lock_guard<std::mutex> lk(lock);
        done = true;
      }
      condition.notify_one();
    }

    void wait()
    {
      std::unique_lock<std::mutex> lk(lock);
      condition.wait(lk, [&done = done]{return done;});
      done = false;
    }

  private:
    std::mutex lock;
    std::condition_variable condition;
    bool done;
};