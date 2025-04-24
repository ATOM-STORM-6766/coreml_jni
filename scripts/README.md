# Core ML Tools

A collection of utility tools for working with Core ML models.

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Tools Overview

### 1. Model Description Viewer (coreml_description.py)

A tool to inspect the input and output structure of Core ML models.

```bash
python coreml_description.py path/to/your/model.mlmodel
```

Output includes:

- Input feature names and types
- Output feature names and types

### 2. Real-time Object Detector (coreml_model_detector.py)

A real-time object detection tool using Core ML models with camera input support.

```bash
python coreml_model_detector.py path/to/your/model.mlmodel
```

Features:

- Real-time camera input support
- Multi-threaded processing for better performance
- Performance metrics display (FPS, processing time, etc.)
- Automatic image pre-processing and post-processing
- Support for macOS (with Neural Engine) and other platforms (CPU only)

### 3. Model Quantization Tool (coreml_model_quantization.py)

A tool for quantizing Core ML models to reduce model size.

```bash
python coreml_model_quantization.py -i input_model.mlmodel -o output_model.mlmodel -b [8|16]
```

Parameters:

- `-i` or `--input`: Input model path (required)
- `-o` or `--output`: Output model path (optional, defaults to input filename with bit-depth suffix)
- `-b` or `--bits`: Quantization precision, either 8 or 16 bits (default: 16)

### 4. YOLO PyTorch to CoreML Export Tool (export_coreml.py)

A tool to export YOLO PyTorch models (.pt) to CoreML format (.mlpackage) using Ultralytics.

```bash
python export_coreml.py --input-file path/to/your/model.pt --output-file path/to/save/model.mlpackage [--nms]
```

Parameters:

- `--input-file`: Path to the input PyTorch model file (required, .pt format)
- `--output-file`: Path to save the exported CoreML model (required, must end with .mlpackage)
- `--nms`: Optional, add Non-Max Suppression (NMS) post-processi

## Important Notes

1. All tools require Core ML Tools and related dependencies to be installed.
2. The object detector automatically utilizes Neural Engine acceleration on macOS.
3. Quantized models may experience slight accuracy loss, validation before use is recommended.
