"""
Model export utilities for converting PyTorch models to ONNX format.
"""
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, Any, Optional

from .models import create_model, load_pretrained

def export_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    input_shape: tuple = (1, 3, 224, 224),
    num_classes: int = 2,
    opset_version: int = 12,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    **kwargs
) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch model checkpoint
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        num_classes: Number of output classes
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes configuration
        **kwargs: Additional arguments for model loading
        
    Returns:
        Path to the exported ONNX model
    """
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},  # batch dimension
            'output': {0: 'batch_size'},
        }
    
    # Create model and load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pretrained(
        str(model_path), 
        device=device,
        num_classes=num_classes,
        **kwargs
    )
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export the model
    torch.onnx.export(
        model,                      # PyTorch model
        dummy_input,                # Model input
        str(output_path),           # Output path
        export_params=True,         # Store the trained parameter weights
        opset_version=opset_version,# ONNX version
        do_constant_folding=True,   # Optimize the model
        input_names=['input'],      # Input tensor name
        output_names=['output'],    # Output tensor name
        dynamic_axes=dynamic_axes,  # Variable length axes
    )
    
    # Verify the exported model
    # (This requires onnx and onnxruntime to be installed)
    try:
        import onnx
        import onnxruntime as ort
        
        # Check that the model is well formed
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Verify model with ONNX Runtime
        ort_session = ort.InferenceSession(str(output_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_session.run(None, ort_inputs)
        
        print(f"Successfully exported ONNX model to {output_path}")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {num_classes} classes")
        
    except ImportError:
        print("ONNX or ONNX Runtime not installed. Skipping model verification.")
    
    return str(output_path)

def convert_checkpoint_to_onnx(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    model_name: str = "species_classifier",
    **kwargs
) -> str:
    """
    Convert a PyTorch checkpoint to ONNX format.
    
    Args:
        checkpoint_dir: Directory containing the model checkpoint
        output_dir: Directory to save the ONNX model
        model_name: Base name for the output model
        **kwargs: Additional arguments for export_to_onnx
        
    Returns:
        Path to the exported ONNX model
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir else checkpoint_dir
    
    # Find the best model checkpoint
    checkpoint_path = None
    if (checkpoint_dir / 'model_best.pth').exists():
        checkpoint_path = checkpoint_dir / 'model_best.pth'
    else:
        # Look for the latest checkpoint
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_path = checkpoints[-1]
    
    if not checkpoint_path or not checkpoint_path.exists():
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir}")
    
    # Set default output path if not provided
    if output_dir is None:
        output_dir = checkpoint_dir
    
    output_path = output_dir / f"{model_name}.onnx"
    
    # Export to ONNX
    return export_to_onnx(
        model_path=checkpoint_path,
        output_path=output_path,
        **kwargs
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint or directory containing checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for ONNX model')
    parser.add_argument('--input-shape', type=int, nargs=4, 
                        default=[1, 3, 224, 224],
                        help='Input shape (batch, channels, height, width)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.checkpoint):
        # If directory is provided, find the best checkpoint
        convert_checkpoint_to_onnx(
            checkpoint_dir=args.checkpoint,
            output_dir=os.path.dirname(args.output) if args.output else None,
            model_name=os.path.splitext(os.path.basename(args.output or 'species_classifier'))[0],
            input_shape=tuple(args.input_shape),
            num_classes=args.num_classes,
            opset_version=args.opset
        )
    else:
        # Export a single checkpoint
        export_to_onnx(
            model_path=args.checkpoint,
            output_path=args.output or 'model.onnx',
            input_shape=tuple(args.input_shape),
            num_classes=args.num_classes,
            opset_version=args.opset
        )
