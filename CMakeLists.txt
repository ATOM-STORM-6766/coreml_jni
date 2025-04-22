cmake_minimum_required(VERSION 3.24)

project(coreml_jni CXX OBJC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc -fobjc-weak")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -framework CoreML -framework Vision -framework Foundation -framework AppKit")
endif()

set(BUILD_SHARED_LIBS ON)

set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_ARCH "osxuniversal")
# also valid: windowsx86-64

# type can be "", "debug", "static", or "staticdebug"
set(OPENCV_TYPE "")

# Download opencv, and save the path
include(FetchContent)
FetchContent_Declare(
    opencv_lib
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-${OPENCV_ARCH}${OPENCV_TYPE}.zip
)
FetchContent_MakeAvailable(opencv_lib)

# download OpenCV headers
FetchContent_Declare(
    opencv_header
    URL https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first/thirdparty/${OPENCV_YEAR}/opencv/opencv-cpp/${OPENCV_VERSION}/opencv-cpp-${OPENCV_VERSION}-headers.zip
)
FetchContent_MakeAvailable(opencv_header)

file(GLOB_RECURSE OPENCV_LIB_PATH
  "${opencv_lib_SOURCE_DIR}/**/*.dylib"
)
list(REMOVE_ITEM OPENCV_LIB_PATH "${opencv_lib_SOURCE_DIR}/osx/universal/shared/libopencv_java4100.dylib")
SET(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("OpenCV libraries found: ${OPENCV_LIB_PATH}")

# Manual find Java and set JNI_INCLUDE_DIRS
# find_package(JNI REQUIRED) just doesn't work on macOS, I don't know why
find_package(Java REQUIRED)
if(Java_FOUND)
    execute_process(
        COMMAND bash -c "java -XshowSettings:properties -version 2>&1 | grep 'java.home' | cut -d '=' -f2 | tr -d '[:space:]'"
        OUTPUT_VARIABLE JAVA_HOME_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    message(STATUS "Found Java home: ${JAVA_HOME_PATH}")
    
    set(JAVA_INCLUDE_PATH "${JAVA_HOME_PATH}/include")
    if(APPLE)
        set(JAVA_INCLUDE_PATH2 "${JAVA_INCLUDE_PATH}/darwin")
    endif()
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

file(GLOB_RECURSE SOURCES
    "src/main/native/objc/*.cpp"
    "src/main/native/objc/*.mm"
)

# Option to control the compile-time log level
# 0: NONE, 1: ERROR, 2: INFO, 3: PERF, 4: DEBUG
set(COMPILE_LOG_LEVEL 2 CACHE STRING "Set compile-time log level (0=NONE, 1=ERROR, 2=INFO, 3=PERF, 4=DEBUG)")
message(STATUS "Compile Log Level set to: ${COMPILE_LOG_LEVEL}")

add_library(${PROJECT_NAME} SHARED
    ${SOURCES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OPENCV_LIB_PATH}
    "-Wl,-rpath,@loader_path"
)

if(APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC
        "-framework CoreML"
        "-framework Vision"
        "-framework Foundation"
        "-framework AppKit"
        "-framework CoreVideo"
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JNI_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_PATH}
)

# Pass the log level definition to the compiler
target_compile_definitions(${PROJECT_NAME} PRIVATE CURRENT_LOG_LEVEL=${COMPILE_LOG_LEVEL})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "@loader_path/"
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
