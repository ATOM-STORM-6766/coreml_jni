## Build Instructions

### Environment Requirements

- macOS
- CMake
- OpenJDK 17

### Development Environment Setup

#### Recommended IDE Setup

- Install [Visual Studio Code](https://code.visualstudio.com/)
- Install the following VS Code extensions:
  - [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
  - [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)

#### Building with VS Code

1. Open the project in VS Code
2. Select the "Default" preset when prompted for CMake configuration
3. Let CMake initialize the configuration
4. To build and install, use the "CMake: Install" command in VS Code
5. Proceed rest of build and publish with Gradle

#### Manual Build Process

1. Install dependencies:

```bash
brew install cmake
```

2. Configure and build the project using CMake:

```bash
cmake -B cmake_build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake_build
```

3. Build and install:

```bash
cmake --build cmake_build --target install -- -j 4
```

4. Build with Gradle:

```bash
./gradlew build
```

5. Publish to local Maven repository:

```bash
./gradlew publishToMavenLocal
```

The built dynamic libraries will be installed to:

- CMake install directory: `cmake_build/lib/`
- Gradle output directory: `build/outputs/nativelibraries/osxuniversal/`

### Using the Artifacts

Since the artifacts are published to GitHub Packages, you need to configure your `build.gradle` or `settings.gradle` to include the GitHub Packages Maven repository. Add the following to your `repositories` block:

```gradle
repositories {
    mavenCentral() // or jcenter()
    maven {
        name = "AtomStorm"
        url = uri("https://maven.pkg.github.com/ATOM-STORM-6766/coreml_jni")
    }
}
```

Then, add the following dependencies to your `build.gradle` file:

```gradle
implementation("org.atomstorm:coreml_jni-jni:$coremlVersion:osxuniversal") {
    transitive = false
}
implementation("org.atomstorm:coreml_jni-java:$coremlVersion") {
    transitive = false
}
```

The `$coremlVersion` should be replaced with the actual version number, which follows a format like `dev-v2025.0.0-10-xxx`. You can find this version number in the terminal output after running `./gradlew publishToMavenLocal`.

Note: Make sure you have published the artifacts to your local Maven repository using `./gradlew publishToMavenLocal` before using them in other projects.

## Utility Scripts

The `scripts/` directory contains several utility tools for working with Core ML models.

For detailed information about these tools and their usage, please refer to [scripts/README.md](scripts/README.md).

## Acknowledgements

This project is adapted from [PhotonVision/rknn_jni](https://github.com/PhotonVision/rknn_jni). We appreciate their work and contribution to the open source community.
