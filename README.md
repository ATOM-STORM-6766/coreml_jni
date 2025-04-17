## 构建说明

### 环境要求

- macOS (arm64)
- CMake
- OpenJDK 17

### 安装依赖

```bash
brew install cmake
```

### 构建步骤

1. 使用 CMake 配置和构建项目

```bash
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake_build
```

2. 构建并安装

```bash
cmake --build cmake_build --target install -- -j 4
```

3. 使用 Gradle 构建

```bash
./gradlew build
```

构建完成后，动态链接库会被安装到以下位置：

- CMake 安装目录：`cmake_build/lib/`
- Gradle 输出目录：`build/outputs/nativelibraries/osxuniversal/`
