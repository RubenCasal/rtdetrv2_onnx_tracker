# trt_inference_v10_fixed.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # crea/destroye contexto CUDA
import numpy as np, cv2, torch, time
from transformers import AutoImageProcessor, AutoConfig
from PIL import Image

class ObjectDetectionOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes

# ---------- CONFIG ----------
ENGINE_PATH = "rt_detr_fp16.engine"
CKPT       = "./ptfue_dataset_pequeño_checkpoint/checkpoint-5000"
VIDEO_IN   = "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/isla_de_lobos_barcos_personas.mp4"
VIDEO_OUT  = "./output_tracking/trt_inference.mp4"

CLASS_THR  = {0: 0.6, 1: 0.4}  # {class_id: score_thr}
CONF_MIN   = float(min(CLASS_THR.values())) if CLASS_THR else 0.5
# ----------------------------

DTYPE_NP = {
    trt.DataType.FLOAT:  np.float32,
    trt.DataType.HALF:   np.float16,
    trt.DataType.INT32:  np.int32,
    trt.DataType.INT8:   np.int8,
    trt.DataType.BOOL:   np.bool_,
}

def nbytes(shape, dtype_np) -> int:
    return int(np.prod(shape)) * np.dtype(dtype_np).itemsize

class DevBuf:
    """Contiene el puntero device y los bytes reservados."""
    def __init__(self):
        self.ptr = None
        self.nbytes = 0
    def free(self):
        if self.ptr is not None:
            try:
                self.ptr.free()
            except Exception:
                pass
            self.ptr = None
            self.nbytes = 0

def ensure_devbuf(devbuf: DevBuf, needed_bytes: int):
    """(Re)aloja si hace falta."""
    if devbuf.ptr is None or devbuf.nbytes < needed_bytes:
        devbuf.free()
        devbuf.ptr = cuda.mem_alloc(needed_bytes)
        devbuf.nbytes = needed_bytes

# ====== Cargar engine TRT y preparar contexto ======
logger  = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
assert engine is not None, "No se pudo deserializar el engine TRT"

context = engine.create_execution_context()
assert context is not None, "No se pudo crear IExecutionContext"

stream = cuda.Stream()

# Descubrir I/O por nombre
inputs, outputs = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)

print(f"[TRT] inputs={inputs} | outputs={outputs}")
assert len(inputs) == 1, f"Se esperaba 1 input y hay {len(inputs)}"
assert len(outputs) >= 2, f"Se esperaban al menos 2 outputs y hay {len(outputs)}"

inp_name = inputs[0]
try:
    logits_name = next(n for n in outputs if "logits" in n)
    boxes_name  = next(n for n in outputs if "pred_boxes" in n or "boxes" in n)
except StopIteration:
    raise RuntimeError(f"No se encuentran outputs 'logits'/'pred_boxes' en {outputs}")

# Tipos del engine
in_dtype_trt    = engine.get_tensor_dtype(inp_name)
logits_dtype_t  = engine.get_tensor_dtype(logits_name)
boxes_dtype_t   = engine.get_tensor_dtype(boxes_name)
in_dtype_np     = DTYPE_NP[in_dtype_trt]
logits_dtype_np = DTYPE_NP[logits_dtype_t]
boxes_dtype_np  = DTYPE_NP[boxes_dtype_t]

# ====== Processor y etiquetas HF ======
processor = AutoImageProcessor.from_pretrained(CKPT)
cfg       = AutoConfig.from_pretrained(CKPT)
id2label  = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()}

# ====== Vídeo IO ======
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), f"No se puede abrir el vídeo: {VIDEO_IN}"

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
assert writer.isOpened(), f"No se puede abrir el writer: {VIDEO_OUT}"

# ====== Buffers device reutilizables ======
d_input  = DevBuf()
d_logits = DevBuf()
d_boxes  = DevBuf()

# ====== Loop de frames ======
frame_idx = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)

    # ---------- PREPROCESADO ----------
    t0 = time.time()
    inputs_pt = processor(images=pil_img, return_tensors="pt")  # [1,3,Hr,Wr]
    pv = inputs_pt["pixel_values"]
    in_shape = tuple(pv.shape)  # p.ej. (1,3,640,640)
    context.set_input_shape(inp_name, in_shape)

    pv_np = pv.detach().cpu().numpy().astype(in_dtype_np, copy=False)

    # Reservar/cargar input
    ensure_devbuf(d_input, nbytes(in_shape, in_dtype_np))
    cuda.memcpy_htod_async(d_input.ptr, pv_np, stream)
    context.set_tensor_address(inp_name, int(d_input.ptr))
    t1 = time.time()

    # ---------- PREPARAR SALIDAS ----------
    log_shape = tuple(context.get_tensor_shape(logits_name))
    box_shape = tuple(context.get_tensor_shape(boxes_name))

    ensure_devbuf(d_logits, nbytes(log_shape, logits_dtype_np))
    ensure_devbuf(d_boxes,  nbytes(box_shape,  boxes_dtype_np))
    context.set_tensor_address(logits_name, int(d_logits.ptr))
    context.set_tensor_address(boxes_name,  int(d_boxes.ptr))

    # ---------- INFERENCIA (v3) ----------
    ok_exec = context.execute_async_v3(stream.handle)
    assert ok_exec, "execute_async_v3() devolvió False"

    logits_np = np.empty(log_shape, dtype=logits_dtype_np)
    boxes_np  = np.empty(box_shape,  dtype=boxes_dtype_np)
    cuda.memcpy_dtoh_async(logits_np, d_logits.ptr, stream)
    cuda.memcpy_dtoh_async(boxes_np,  d_boxes.ptr,  stream)
    stream.synchronize()
    t2 = time.time()

    # ---------- POSTPROCESADO (HF) ----------
    outputs_hf = ObjectDetectionOutput(
        logits=torch.from_numpy(logits_np).float(),     # [1,Q,C]
        pred_boxes=torch.from_numpy(boxes_np).float(),  # [1,Q,4]
    )
    res = processor.post_process_object_detection(
        outputs_hf, threshold=CONF_MIN, target_sizes=[(H, W)]
    )[0]

    scores = res["scores"]
    labels = res["labels"].to(torch.long)
    boxes  = res["boxes"]

    if scores.numel() > 0 and CLASS_THR:
        num_classes = max(int(labels.max().item()) + 1, max(CLASS_THR.keys()) + 1)
        thr_table = torch.full((num_classes,), CONF_MIN, dtype=torch.float32)
        idx  = torch.tensor(list(CLASS_THR.keys()), dtype=torch.long)
        vals = torch.tensor(list(CLASS_THR.values()), dtype=torch.float32)
        thr_table.scatter_(0, idx, vals)
        mask = scores >= thr_table[labels]
        scores, labels, boxes = scores[mask], labels[mask], boxes[mask]
    t3 = time.time()

    # ---------- DIBUJO ----------
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, b.tolist())
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        name = id2label.get(int(l), str(int(l)))
        cv2.putText(frame_bgr, f"{name}:{float(s):.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame_bgr)
    frame_idx += 1

    # ---------- PRINT TIEMPOS ----------
    pre_t   = (t1 - t0) * 1000.0
    inf_t   = (t2 - t1) * 1000.0
    post_t  = (t3 - t2) * 1000.0
    total_t = (t3 - t0) * 1000.0
    print("###########")
    print(f"Frame {frame_idx:04d} | Pre: {pre_t:.2f} ms | TRT: {inf_t:.2f} ms | Post: {post_t:.2f} ms | Total: {total_t:.2f} ms")
    print("###########")

# ====== Limpieza ======
cap.release()
writer.release()
d_input.free(); d_logits.free(); d_boxes.free()
print(f"[OK] Vídeo guardado: {VIDEO_OUT}")
