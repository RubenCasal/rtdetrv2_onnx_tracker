import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# === CONFIG ===
CHECKPOINT = "./ptfue_dataset_pequeño_checkpoint/checkpoint-5000"
ONNX_PATH = "rt_detr_fixed.onnx"
# ===============

# 1️⃣ Cargar modelo y processor
model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT, use_safetensors=True)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
model.eval()

# 2️⃣ Crear wrapper para exponer salidas explícitas
class RTDETRWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes  # <- nombres claros

# 3️⃣ Exportar con I/O correctos
wrapped = RTDETRWrapper(model)
dummy = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    wrapped,
    (dummy,),
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    },
    opset_version=17,
)

print(f"✅ Modelo ONNX exportado correctamente a {ONNX_PATH}")
