// First include C/C++ headers
#include <CoreVideo/CVPixelBuffer.h>
#include <cstddef>

// Then include our headers
#import "PixelBufferPool.h"
#import "Log.h"

// Maximum number of buffers to keep in the pool for each size
#define MAX_BUFFERS_PER_SIZE 5

@interface PixelBufferPool ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableArray<NSValue *> *> *bufferPool;
@property (nonatomic, strong) dispatch_queue_t queue;
@end

@implementation PixelBufferPool

+ (instancetype)sharedPool {
    static PixelBufferPool *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[PixelBufferPool alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _bufferPool = [NSMutableDictionary dictionary];
        _queue = dispatch_queue_create("com.pixelbufferpool.queue", DISPATCH_QUEUE_SERIAL);
        LOG_INFO("PixelBufferPool initialized");
    }
    return self;
}

- (void)dealloc {
    [self clear];
    LOG_INFO("PixelBufferPool destroyed");
}

- (NSString *)keyForWidth:(size_t)width height:(size_t)height {
    return [NSString stringWithFormat:@"%zu-%zu", width, height];
}

- (CVPixelBufferRef)createPixelBufferWithWidth:(size_t)width height:(size_t)height {
    NSDictionary *pixelBufferAttributes = @{
        (NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferWidthKey: @(width),
        (NSString *)kCVPixelBufferHeightKey: @(height),
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };

    CVPixelBufferRef buffer = nullptr;
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)pixelBufferAttributes,
        &buffer
    );

    if (status != kCVReturnSuccess) {
        LOG_ERROR("Failed to create pixel buffer: %d", status);
        return nullptr;
    }

    return buffer;
}

- (CVPixelBufferRef)getPixelBufferWithWidth:(size_t)width height:(size_t)height {
    if (width <= 0 || height <= 0) {
        LOG_ERROR("Invalid dimensions: %zu x %zu", width, height);
        return nullptr;
    }

    NSString *key = [self keyForWidth:width height:height];
    __block CVPixelBufferRef buffer = nullptr;

    // Try to get a buffer from the pool
    dispatch_sync(_queue, ^{
        NSMutableArray *buffers = self.bufferPool[key];
        if (buffers.count > 0) {
            buffer = (CVPixelBufferRef)[buffers.lastObject pointerValue];
            [buffers removeLastObject];
            LOG_DEBUG("Reusing pixel buffer %p (%zu x %zu)", buffer, width, height);
        }
    });

    // If no buffer was available, create a new one
    if (!buffer) {
        buffer = [self createPixelBufferWithWidth:width height:height];
        if (buffer) {
            LOG_DEBUG("Created new pixel buffer %p (%zu x %zu)", buffer, width, height);
        }
    }

    return buffer;
}

- (void)returnPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!pixelBuffer) {
        return;
    }

    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    NSString *key = [self keyForWidth:width height:height];

    __block BOOL shouldRelease = NO;

    dispatch_sync(_queue, ^{
        NSMutableArray *buffers = self.bufferPool[key];
        if (!buffers) {
            buffers = [NSMutableArray array];
            self.bufferPool[key] = buffers;
        }

        // Only keep a limited number of buffers per size
        if (buffers.count < MAX_BUFFERS_PER_SIZE) {
            [buffers addObject:[NSValue valueWithPointer:pixelBuffer]];
            LOG_DEBUG("Returned pixel buffer %p to pool (%zu x %zu)", pixelBuffer, width, height);
        } else {
            shouldRelease = YES;
            LOG_DEBUG("Pool full for size %zu x %zu, releasing buffer %p", width, height, pixelBuffer);
        }
    });

    if (shouldRelease) {
        CVPixelBufferRelease(pixelBuffer);
    }
}

- (void)clear {
    dispatch_sync(_queue, ^{
        for (NSString *key in self.bufferPool) {
            NSMutableArray *buffers = self.bufferPool[key];
            for (NSValue *value in buffers) {
                CVPixelBufferRef buffer = (CVPixelBufferRef)[value pointerValue];
                CVPixelBufferRelease(buffer);
            }
            [buffers removeAllObjects];
        }
        [self.bufferPool removeAllObjects];
        LOG_INFO("Pixel buffer pool cleared");
    });
}

@end
