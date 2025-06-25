import argparse
import torch
import os

# Try to import torch2trt, fallback to ONNX+TensorRT if not available
try:
    from torch2trt import torch2trt, TRTModule
    TORCH2TRT_AVAILABLE = True
except ImportError:
    TORCH2TRT_AVAILABLE = False

# Optional: ONNX+TensorRT fallback
try:
    import onnx
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    ONNX_TRT_AVAILABLE = True
except ImportError:
    ONNX_TRT_AVAILABLE = False

def load_model(pt_path):
    model = torch.load(pt_path)
    if hasattr(model, 'eval'):
        model.eval()
    return model

def convert_with_torch2trt(model, example_input, precision, output_path):
    print(f"Converting with torch2trt, precision: {precision}")
    fp16_mode = precision == 'fp16'
    int8_mode = precision == 'int8'
    model_trt = torch2trt(
        model,
        [example_input],
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        max_batch_size=example_input.shape[0],
    )
    torch.save(model_trt.state_dict(), output_path)
    print(f"TensorRT model saved to {output_path}")

def convert_with_onnx_trt(model, example_input, precision, output_path):
    print(f"Converting with ONNX+TensorRT, precision: {precision}")
    onnx_path = output_path.replace('.trt', '.onnx')
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print(f"ONNX model exported to {onnx_path}")
    # Build TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # For real int8, calibration is needed. Here we just set the flag for demo.
    engine = builder.build_engine(network, config)
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt model to TensorRT format.")
    parser.add_argument('pt_path', type=str, help='Path to the .pt model file')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32', help='Precision mode for TensorRT')
    parser.add_argument('--output', type=str, default='model_trt.pth', help='Output path for TensorRT model/engine')
    parser.add_argument('--input-shape', type=int, nargs='+', required=True, help='Input shape, e.g. --input-shape 1 3 224 224')
    args = parser.parse_args()

    model = load_model(args.pt_path)
    example_input = torch.randn(*args.input_shape).cuda() if torch.cuda.is_available() else torch.randn(*args.input_shape)

    if TORCH2TRT_AVAILABLE:
        convert_with_torch2trt(model, example_input, args.precision, args.output)
    elif ONNX_TRT_AVAILABLE:
        convert_with_onnx_trt(model, example_input, args.precision, args.output)
    else:
        print("Neither torch2trt nor ONNX+TensorRT are available. Please install one of them.")

if __name__ == '__main__':
    main()
