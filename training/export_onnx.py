"""Export fine-tuned model to ONNX and apply INT8 quantization.

Usage:
    python training/export_onnx.py --model models/finetuned --output models/onnx
"""

import argparse
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime.quantization import ORTQuantizer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/finetuned")
    parser.add_argument("--output", type=str, default="models/onnx")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_path} to ONNX...")

    # Step 1: Export to ONNX
    onnx_model = ORTModelForTokenClassification.from_pretrained(
        model_path, export=True
    )
    onnx_export_dir = output_path / "onnx_export"
    onnx_model.save_pretrained(str(onnx_export_dir))

    # Save tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(str(onnx_export_dir))

    print(f"ONNX model exported to {onnx_export_dir}")

    # Step 2: Quantize (dynamic INT8)
    print("Applying dynamic INT8 quantization...")
    quantizer = ORTQuantizer.from_pretrained(onnx_export_dir)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    quantized_dir = output_path / "quantized"
    quantizer.quantize(save_dir=str(quantized_dir), quantization_config=qconfig)

    # Copy tokenizer files to quantized dir
    tokenizer.save_pretrained(str(quantized_dir))

    # Copy label map if present
    label_map = model_path / "label_map.json"
    if label_map.exists():
        shutil.copy(label_map, quantized_dir / "label_map.json")

    print(f"Quantized model saved to {quantized_dir}")

    # Report sizes
    onnx_size = sum(f.stat().st_size for f in onnx_export_dir.glob("*.onnx"))
    quant_size = sum(f.stat().st_size for f in quantized_dir.glob("*.onnx"))
    print(f"\nONNX model size: {onnx_size / 1e6:.1f} MB")
    print(f"Quantized model size: {quant_size / 1e6:.1f} MB")
    print(f"Size reduction: {(1 - quant_size / onnx_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
