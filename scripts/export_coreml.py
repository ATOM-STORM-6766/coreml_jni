import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Export a YOLO model (.pt) to CoreML format (.mlmodel) using ultralytics."
    )
    parser.add_argument(
        "--input-file", type=Path, help="Path to the input PyTorch model file (.pt)."
    )
    parser.add_argument(
        "--output-file", type=Path, help="Path to save the exported CoreML model file (.mlmodel)."
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Add Non-Max Suppression (NMS) to the exported CoreML model.",
    )
    args = parser.parse_args()

    input_file: Path = args.input_file
    output_file: Path = args.output_file

    if not input_file.is_file():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    if output_file.suffix.lower() != ".mlmodel":
        print(
            f"Error: Output file must have a .mlmodel extension. Got: '{output_file}'"
        )
        sys.exit(1)

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading model from '{input_file}' and exporting to CoreML format..."
    )

    try:
        # Load the YOLO model
        model = YOLO(input_file)

        # Export the model to CoreML format
        # The export function directly returns the path to the exported model
        # or saves it based on internal logic if no specific path handling is needed upfront.
        # Ultralytics handles naming and saving, but we want a specific output path.

        # Perform the export. Ultralytics might save it to a default location first.
        # We'll handle moving/renaming if necessary after export.
        model.export(format="mlmodel", nms=args.nms) # Export with or without NMS based on arg

        # Ultralytics>=8.0.131 typically saves the exported model in the same directory
        # as the input model, with the name like 'yolov8n_coreml_model/yolov8n.mlmodel'.
        # Or sometimes just 'yolov8n.mlmodel' in the CWD or a specific export dir.
        # We need to find the exported file.
        input_stem = input_file.stem
        expected_export_dir = Path.cwd() / f"{input_stem}_coreml_model"
        expected_export_file = expected_export_dir / f"{input_stem}.mlmodel"

        # Alternative potential location (newer ultralytics might save directly)
        alt_export_file = Path.cwd() / f"{input_stem}.mlmodel"

        exported_file_path = None
        if expected_export_file.is_file():
             exported_file_path = expected_export_file
        elif alt_export_file.is_file():
             exported_file_path = alt_export_file
        else:
            # Search common locations if the predictable paths don't work
            possible_files = list(Path.cwd().glob(f"**/{input_stem}.mlmodel"))
            if possible_files:
                exported_file_path = possible_files[0] # Take the first match
                print(f"Found exported model at: {exported_file_path}")
            else:
                 print("Error: Could not automatically find the exported .mlmodel file.")
                 print(f"Looked in: {expected_export_file}, {alt_export_file}, and CWD subdirs")
                 sys.exit(1)

        # Move the exported file to the desired output location
        exported_file_path.rename(output_file)
        print(f"Successfully moved exported model to '{output_file}'")
        print(f"Export complete: '{output_file}'")

    except Exception as e:
        print(f"An error occurred during export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 