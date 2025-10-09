# onnx_inference_video_timed_iobinding.py
import cv2, onnxruntime as ort, torch, time, numpy as np
from transformers import AutoImageProcessor, AutoConfig
from PIL import Image

class ObjectDetectionOutput:
    def __init__(self, logits, pred_boxes):
        self.logits = logits
        self.pred_boxes = pred_boxes

CKPT = "./ptfue_dataset_pequeño_checkpoint/checkpoint-5000"
ONNX = "rt_detr_fixed.onnx"          # si tienes un .fp16, cambia aquí
SRC_VIDEO = "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/isla_de_lobos_barcos_personas.mp4"
DST_VIDEO = "./output_tracking/output_annotated_onnx.mp4"
CLASS_THR = {0: 0.6, 1: 0.4, 2:0.8}
CONF_MIN = float(min(CLASS_THR.values())) if CLASS_THR else 0.5

# --- Configuración de precisión ---
USE_FP16 = False  # pon True si tu modelo ONNX es FP16

# --- Crear sesión ORT con optimizaciones + CUDA provider options ---
use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")

cuda_provider_options = {
    "cudnn_conv_use_max_workspace": "1",
    "do_copy_in_default_stream": "1",

}

providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
sess = ort.InferenceSession(ONNX, sess_options=sess_opts, providers=providers)

# I/O names
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
assert out_names == ["logits", "pred_boxes"], f"Salidas inesperadas: {out_names}"

# Processor + labels
processor = AutoImageProcessor.from_pretrained(CKPT)
cfg = AutoConfig.from_pretrained(CKPT)
id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()}

# Video IO
cap = cv2.VideoCapture(SRC_VIDEO)
fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(DST_VIDEO, fourcc, fps, (W, H))

# Utilidades para IOBinding
device_id = 0
dtype_np  = np.float16 if USE_FP16 else np.float32
dtype_t   = torch.float16 if USE_FP16 else torch.float32

# (opcional) warmup para compilar CUDA Graph/heurísticas
def warmup_io_binding(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    pv_t = inputs["pixel_values"].to("cuda" if use_cuda else "cpu", non_blocking=True).to(dtype_t)
    io = sess.io_binding()
    # Bind input en GPU (evita copia intermedia)
    io.bind_input(
        name=in_name, device_type="cuda" if use_cuda else "cpu", device_id=device_id if use_cuda else 0,
        element_type=dtype_np, shape=list(pv_t.shape), buffer_ptr=pv_t.data_ptr() if use_cuda else None
    )
    # Bind outputs en GPU (ORT los aloja)
    for o in out_names:
        io.bind_output(o, "cuda" if use_cuda else "cpu", device_id if use_cuda else 0)
    sess.run_with_iobinding(io)
    io.copy_outputs_to_cpu()

# Warmup con un frame (si existe)
ret, warm_frame = cap.read()
if ret:
    warm_pil = Image.fromarray(cv2.cvtColor(warm_frame, cv2.COLOR_BGR2RGB))
    warmup_io_binding(warm_pil)
    # rebobinar al principio para procesar todo el vídeo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok: break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # ---------- PREPROCESADO ----------
    t0 = time.time()
    inputs = processor(images=pil_img, return_tensors="pt")
    # tensor en GPU/CPU según EP; aquí lo ponemos en GPU para IOBinding
    pv_t = inputs["pixel_values"].to("cuda" if use_cuda else "cpu", non_blocking=True).to(dtype_t)
    t1 = time.time()

    # ---------- INFERENCIA con IOBinding (sin copias CPU<->GPU) ----------
    io = sess.io_binding()
    io.bind_input(
        name=in_name,
        device_type="cuda" if use_cuda else "cpu",
        device_id=device_id if use_cuda else 0,
        element_type=dtype_np,
        shape=list(pv_t.shape),
        buffer_ptr=pv_t.data_ptr() if use_cuda else None,
    )
    for o in out_names:
        io.bind_output(o, "cuda" if use_cuda else "cpu", device_id if use_cuda else 0)

    sess.run_with_iobinding(io)
    # Recoger salidas a CPU para postproceso HF (devuelve numpy)
    logits_np, boxes_np = io.copy_outputs_to_cpu()
    t2 = time.time()

    # ---------- POSTPROCESADO ----------
    outputs = ObjectDetectionOutput(
        logits=torch.from_numpy(logits_np),       # [1,Q,C]
        pred_boxes=torch.from_numpy(boxes_np),    # [1,Q,4]
    )
    res = processor.post_process_object_detection(
        outputs, threshold=CONF_MIN, target_sizes=[(H, W)]
    )[0]
    scores = res["scores"]
    labels = res["labels"].to(torch.long)
    boxes  = res["boxes"]

    if scores.numel() > 0 and CLASS_THR:
        num_classes = max(int(labels.max().item()) + 1, max(CLASS_THR.keys()) + 1)
        thr_table = torch.full((num_classes,), CONF_MIN, dtype=torch.float32)
        idx = torch.tensor(list(CLASS_THR.keys()), dtype=torch.long)
        vals = torch.tensor(list(CLASS_THR.values()), dtype=torch.float32)
        thr_table.scatter_(0, idx, vals)
        mask = scores >= thr_table[labels]
        scores, labels, boxes = scores[mask], labels[mask], boxes[mask]
    t3 = time.time()

    # ---------- DIBUJO ----------
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        name = id2label.get(int(l), str(int(l)))
        cv2.putText(frame_bgr, f"{name}:{float(s):.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    writer.write(frame_bgr)
    frame_idx += 1

    # ---------- PRINT TIEMPOS ----------
    pre_t   = (t1 - t0) * 1000
    infer_t = (t2 - t1) * 1000
    post_t  = (t3 - t2) * 1000
    total_t = (t3 - t0) * 1000
    print("###########")
    print(f"Frame {frame_idx:04d} | Pre: {pre_t:.2f} ms | Inference: {infer_t:.2f} ms | Post: {post_t:.2f} ms | Total: {total_t:.2f} ms")
    print("###########")

cap.release()
writer.release()
print(f"[OK] Guardado: {DST_VIDEO}")
