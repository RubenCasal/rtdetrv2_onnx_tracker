import tensorrt as trt, sys

ONNX = "rt_detr_fixed.onnx"
ENGINE = "rt_detr_fp16.engine"
INPUT_SHAPE = (1, 3, 640, 640)

logger = trt.Logger(trt.Logger.INFO)

def build():
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    # workspace 4 GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # FP16 si la GPU lo soporta
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX parse errors:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    # fuerza shape fijo en la 1ª entrada
    inp = network.get_input(0)
    inp.shape = INPUT_SHAPE

    # API moderna: devuelve bytes serializados del plan
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, "build_serialized_network devolvió None"

    with open(ENGINE, "wb") as f:
        f.write(plan)
    print(f"[OK] guardado: {ENGINE}")

if __name__ == "__main__":
    build()
