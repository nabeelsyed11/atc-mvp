#!/usr/bin/env python3
"""
Script to export a trained PyTorch model to ONNX format.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ssl.export import export_to_onnx, convert_checkpoint_to_onnx

def parse_args():
    parser = argparse.ArgumentParser(description='Export a trained model to ONNX format')
    
    # Required arguments
    parser.add_argument('checkpoint', type=str,
                        help='Path to the PyTorch checkpoint or directory containing checkpoints')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the ONNX model')
    parser.add_argument('--input-shape', type=int, nargs=4, 
                        default=[1, 3, 224, 224],
                        help='Input shape as batch, channels, height, width (default: 1 3 224 224)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_dir():
            output_path = checkpoint_path / 'model.onnx'
        else:
            output_path = checkpoint_path.with_suffix('.onnx')
    else:
        output_path = args.output
    
    # Export the model
    if os.path.isdir(args.checkpoint):
        output_path = convert_checkpoint_to_onnx(
            checkpoint_dir=args.checkpoint,
            output_path=output_path,
            input_shape=tuple(args.input_shape),
            num_classes=args.num_classes,
            opset_version=args.opset
        )
    else:
        output_path = export_to_onnx(
            model_path=args.checkpoint,
            output_path=output_path,
            input_shape=tuple(args.input_shape),
            num_classes=args.num_classes,
            opset_version=args.opset
        )
    
    print(f"\nModel successfully exported to: {output_path}")
    print("\nYou can now use this ONNX model with the existing inference pipeline.")
    print("Update the SPECIES_ONNX_PATH in your .env file to point to this model.")

if __name__ == "__main__":
    main()
