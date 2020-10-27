#include "reduce.h"

#include <stdint.h>
#include <stdexcept>

namespace tarantella
{
  namespace collectives
  {
    namespace 
    {
      template <class T>
      inline void add(const reduce::task& t) {
        const T* const a = (const T*) t.source;
        T* const b = (T*) t.destination;
        const long n = t.len;

        for (long i=0; i < n; i++) {
          b[i] += a[i];
        }
    }

    template <class T>
    inline void average(const reduce::task& t) {
      if (t.scaling > 1) {
        const T* const a = (const T*) t.source;
        T* const b = (T*) t.destination;
        const long n = t.len;
        const T s = t.scaling;

        for (long i=0; i < n; i++) {
          b[i] = (b[i] + a[i]) / s;
        }
      } else {
        add<T>(t);
      }
    }

    template <class T>
    inline void averageopt(const reduce::task& t) {
      if (t.scaling > 1) {
        const T* const a = (const T*) t.source;
        T* const b = (T*) t.destination;
        const long n = t.len;
        const T s = T(1) / T(t.scaling);

        for (long i=0; i < n; i++) {
          b[i] = (b[i] + a[i]) * s;
        }
      } else {
        add<T>(t);
      }
    }

    class reduce_float_sum : public reduce {
    public:
      void operator()(const task& t) const {
        add<float>(t);
      }
    };

    class reduce_float_average : public reduce {
    public:
      void operator()(const task& t) const {
        averageopt<float>(t);
      }
    };

    class reduce_double_sum : public reduce {
    public:
      void operator()(const task& t) const {
        add<double>(t);
      }
    };

    class reduce_double_average : public reduce {
    public:
      void operator()(const task& t) const {
        averageopt<double>(t);
      }
    };

    class reduce_int16_sum : public reduce {
    public:
      void operator()(const task& t) const {
        add<int16_t>(t);
      }
    };

    class reduce_int16_average : public reduce {
    public:
      void operator()(const task& t) const {
        average<int16_t>(t);
      }
    };

    class reduce_int32_sum : public reduce {
    public:
      void operator()(const task& t) const {
        add<int32_t>(t);
      }
    };

    class reduce_int32_average : public reduce {
    public:
      void operator()(const task& t) const {
        average<int32_t>(t);
      }
    };
    }

    reduce * getReduce(const allreduce::dataType data,
                      const allreduce::reductionType reduction) {
      reduce* p = NULL;

      switch (data) {
      case allreduce::FLOAT:
        switch (reduction) {
        case allreduce::SUM:
          p = new reduce_float_sum();
          break;
        case allreduce::AVERAGE:
          p = new reduce_float_average();
          break;
        default:
          break;
        }
        break;
      case allreduce::DOUBLE:
        switch (reduction) {
        case allreduce::SUM:
          p = new reduce_double_sum;
          break;
        case allreduce::AVERAGE:
          p = new reduce_double_average;
          break;
        default:
          break;
        }
        break;
      case allreduce::INT16:
        switch (reduction) {
        case allreduce::SUM:
          p = new reduce_int16_sum;
          break;
        case allreduce::AVERAGE:
          p = new reduce_int16_average;
          break;
        default:
          break;
        }
        break;
      case allreduce::INT32:
        switch (reduction) {
        case allreduce::SUM:
          p = new reduce_int32_sum;
          break;
        case allreduce::AVERAGE:
          p = new reduce_int32_average;
          break;
        default:
          break;
        }
        break;
      default:
        break;
      };

      if (p == NULL) {
        throw std::runtime_error(
          "Unsupported combination of data type and reduction type");
      }

      return p;
    }

    size_t getDataTypeSize(const allreduce::dataType d) {
      const size_t sizes[allreduce::NUM_TYPE] = {
        sizeof(float),
        sizeof(double),
        sizeof(int16_t),
        sizeof(int32_t)
      };

      return sizes[d];
    }
  }
}
