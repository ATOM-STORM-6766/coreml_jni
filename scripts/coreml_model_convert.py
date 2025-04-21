import coremltools as ct
import argparse
import os
import sys

def quantize_model(input_path, output_path, nbits):
    try:
        # Load the original model
        print(f"Loading model: {input_path}")
        model = ct.models.MLModel(input_path)
        
        # Quantize the model
        print(f"Quantizing with {nbits}-bit precision...")
        model_quantized = ct.models.neural_network.quantization_utils.quantize_weights(model, nbits=nbits)
        
        # Save the quantized model
        print(f"Saving quantized model to: {output_path}")
        model_quantized.save(output_path)
        
        print("Done!")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='CoreML Model Quantization Tool')
    parser.add_argument('--input', '-i', required=True, help='Input model path (.mlmodel)')
    parser.add_argument('--output', '-o', help='Output model path (.mlmodel)')
    parser.add_argument('--bits', '-b', type=int, choices=[8, 16], default=16, 
                        help='Quantization precision (8 or 16 bits, default: 16)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1
    
    # Create default output path if not specified
    if not args.output:
        filename, ext = os.path.splitext(args.input)
        args.output = f"{filename}_FP{args.bits}{ext}"
    
    # Execute quantization
    success = quantize_model(args.input, args.output, args.bits)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())