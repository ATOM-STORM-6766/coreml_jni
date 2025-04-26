import argparse
import coremltools as ct
import sys

def print_model_description(model_path):
    """
    Loads a Core ML model and prints its input and output descriptions.

    Args:
        model_path (str): Path to the Core ML model file (.mlmodel or .mlpackage).
    """
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model = ct.models.MLModel(model_path)
        print("Model loaded successfully.")

        # Get the model description
        description = model.get_spec().description

        # Print input descriptions
        print("\nInput Descriptions:")
        if not description.input:
             print("  No inputs found.")
        for feature in description.input:
            print(f"  Name: {feature.name}")
            print(f"  Type: {feature.type}")
            print("  ---")

        # Print output descriptions
        print("\nOutput Descriptions:")
        if not description.output:
            print("  No outputs found.")
        for feature in description.output:
            print(f"  Name: {feature.name}")
            print(f"  Type: {feature.type}")
            print("  ---")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Core ML model input and output descriptions.")
    parser.add_argument("model_path", help="Path to the Core ML model file (.mlmodel or .mlpackage)")

    args = parser.parse_args()

    print_model_description(args.model_path)
