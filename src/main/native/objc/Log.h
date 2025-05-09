#pragma once

#import <Foundation/Foundation.h>

// Define Log Levels if not already defined
#ifndef LOG_LEVEL_NONE
#define LOG_LEVEL_NONE  0
#endif

#ifndef LOG_LEVEL_ERROR
#define LOG_LEVEL_ERROR 1
#endif

#ifndef LOG_LEVEL_INFO
#define LOG_LEVEL_INFO  2
#endif

#ifndef LOG_LEVEL_PERF
#define LOG_LEVEL_PERF  3
#endif

#ifndef LOG_LEVEL_DEBUG
#define LOG_LEVEL_DEBUG 4
#endif

// Set default log level if not defined by compiler/CMake
#ifndef CURRENT_LOG_LEVEL
#define CURRENT_LOG_LEVEL 2 // Default to INFO level
#endif

// Conditional Logging Macros
#ifndef LOG_ERROR
#if CURRENT_LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...) NSLog((@"[ERROR][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) do {} while(0)
#endif
#endif

#ifndef LOG_INFO
#if CURRENT_LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...) NSLog((@"[INFO][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) do {} while(0)
#endif
#endif

#ifndef LOG_PERF
#if CURRENT_LOG_LEVEL >= LOG_LEVEL_PERF
#define LOG_PERF(fmt, ...) NSLog((@"[PERF][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_PERF(fmt, ...) do {} while(0)
#endif
#endif

#ifndef LOG_DEBUG
#if CURRENT_LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) NSLog((@"[DEBUG][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) do {} while(0)
#endif
#endif
