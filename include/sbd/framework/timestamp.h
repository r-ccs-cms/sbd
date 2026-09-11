/// This is a part of sbd
/**
@file /sbd/framework/timestamp.h
@brief time stamp
*/
#ifndef SBD_FRAMEWORK_TIMESTAMP_H
#define SBD_FRAMEWORK_TIMESTAMP_H

#include <chrono>
#include <ctime>
#include <cstdint>

namespace sbd {
  uint64_t make_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&in_time_t);
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;
    uint64_t stamp =
      (static_cast<int64_t>((tm.tm_year + 1900)%100) * 10000000000000000ULL) +
      (static_cast<int64_t>(tm.tm_mon + 1)    * 100000000000000ULL) +
      (static_cast<int64_t>(tm.tm_mday)       * 1000000000000ULL) +
      (static_cast<int64_t>(tm.tm_hour)       * 10000000000ULL) +
      (static_cast<int64_t>(tm.tm_min)        * 100000000ULL) +
      (static_cast<int64_t>(tm.tm_sec)        * 1000000ULL) +
      micros.count();
    return stamp;
  }
}

#endif
