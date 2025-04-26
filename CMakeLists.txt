cmake_minimum_required(VERSION 3.24)

project(coreml_jni CXX OBJC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
endif()

set(BUILD_SHARED_LIBS ON)

set(OPENCV_YEAR "frc2025")
set(OPENCV_VERSION "4.10.0-3")
set(OPENCV_ARCH "osxuniversal")

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


set(FOUND_JNI FALSE)
find_package(JNI)
if(JNI_FOUND)
    message(STATUS "JNI found by CMake: ${JNI_INCLUDE_DIRS}")
    set(FOUND_JNI TRUE)
else()
    message(STATUS "CMake didn't find JNI, trying to find it manually")
    find_package(Java REQUIRED)
    if(Java_FOUND)
        execute_process(
            COMMAND bash -c "java -XshowSettings:properties -version 2>&1 | grep 'java.home' | cut -d '=' -f2 | tr -d '[:space:]'"
            OUTPUT_VARIABLE JAVA_HOME_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE JAVA_CMD_RESULT
        )
        if(JAVA_CMD_RESULT EQUAL 0 AND NOT JAVA_HOME_PATH STREQUAL "")
            message(STATUS "Found Java home: ${JAVA_HOME_PATH}")
            set(JAVA_INCLUDE_PATH "${JAVA_HOME_PATH}/include")
            if(APPLE)
                set(JAVA_INCLUDE_PATH2 "${JAVA_INCLUDE_PATH}/darwin")
            endif()
            if(EXISTS "${JAVA_INCLUDE_PATH}" AND (NOT APPLE OR EXISTS "${JAVA_INCLUDE_PATH2}"))
                set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
                set(FOUND_JNI TRUE)
                message(STATUS "JNI includes found manually: ${JNI_INCLUDE_DIRS}")
            else()
                message(WARNING "Java include directories not found at expected locations")
            endif()
        else()
            message(WARNING "Failed to get Java home path using command line method")
        endif()
    endif()
endif()

if(NOT FOUND_JNI)
    message(FATAL_ERROR "Could not find JNI includes using either CMake or manual methods")
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

# Add Objective-C ARC compile options directly to the target
if(APPLE)
    target_compile_options(${PROJECT_NAME} PRIVATE "-fobjc-arc")
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "@loader_path/"
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
