#pragma once

// First include C/C++ headers
#include <CoreVideo/CVPixelBuffer.h>

// Then include Objective-C headers
#import <Foundation/Foundation.h>

/**
 * A pool for reusing CVPixelBuffer objects to reduce memory allocation overhead.
 * This class manages a cache of pixel buffers organized by size.
 */
@interface PixelBufferPool : NSObject

/**
 * Get the singleton instance of the pixel buffer pool.
 * @return The shared pool instance.
 */
+ (instancetype)sharedPool;

/**
 * Get a pixel buffer from the pool with the specified dimensions.
 * If no suitable buffer is available in the pool, a new one will be created.
 *
 * @param width The width of the pixel buffer in pixels.
 * @param height The height of the pixel buffer in pixels.
 * @return A pixel buffer reference, or nullptr if creation failed.
 */
- (CVPixelBufferRef)getPixelBufferWithWidth:(size_t)width height:(size_t)height;

/**
 * Return a pixel buffer to the pool for reuse.
 * If the pool for this size is full, the buffer will be released.
 *
 * @param pixelBuffer The pixel buffer to return to the pool.
 */
- (void)returnPixelBuffer:(CVPixelBufferRef)pixelBuffer;

/**
 * Clear all cached pixel buffers from the pool.
 */
- (void)clear;

@end
