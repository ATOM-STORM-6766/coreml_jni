cmake_minimum_required(VERSION 3.24)

project(coreml_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 Objective-C++ 编译选项
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

# 获取 OpenCV 库文件路径
file(GLOB_RECURSE OPENCV_LIB_PATH
  "${opencv_lib_SOURCE_DIR}/**/*.dylib"
)
list(REMOVE_ITEM OPENCV_LIB_PATH "${opencv_lib_SOURCE_DIR}/osx/universal/shared/libopencv_java4100.dylib")
SET(OPENCV_INCLUDE_PATH ${opencv_header_SOURCE_DIR})
message("OpenCV libraries found: ${OPENCV_LIB_PATH}")

if(APPLE)
    set(JAVA_HOME "$ENV{JAVA_HOME}")
    set(JAVA_INCLUDE_PATH "${JAVA_HOME}/include")
    set(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/darwin")
    set(JAVA_AWT_LIBRARY "${JAVA_HOME}/lib/libjawt.dylib")
    set(JAVA_AWT_INCLUDE_PATH "${JAVA_HOME}/include/darwin")
endif()

find_package(JNI REQUIRED)
if(JNI_FOUND)
    # Fixes odd AWT dependency
    set(JNI_INCLUDE_DIRS ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2})
    message(STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
endif()

# 添加源文件
file(GLOB_RECURSE SOURCES
    "src/main/native/objc/*.cpp"
    "src/main/native/objc/*.mm"
)

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

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        INSTALL_RPATH "@loader_path/"
)

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME})
