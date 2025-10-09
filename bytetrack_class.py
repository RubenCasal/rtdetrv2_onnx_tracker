import os
import cv2
import numpy as np
from collections import defaultdict
import supervision as sv
import torch
import time
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from supervision import ByteTrack as SVByteTrack


class ByteTrackRTDETRv2Stable:
    def __init__(
        self,
        ckpt_path: str,
        video_paths: list[str],
        classes_of_interest_ids: list[int],
        class_id_to_name: dict[int, str],
        det_conf_by_id: dict[int, float],
        tracker_minimum_consecutive_frames: int,
        max_time_lost: int,
      
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForObjectDetection.from_pretrained(ckpt_path).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(ckpt_path, use_fast=True)
        self.det_conf_by_id = det_conf_by_id

        self.video_paths = video_paths
        self.classes = set(int(i) for i in classes_of_interest_ids)
        self.id2name = dict(class_id_to_name)

        self.tracker = SVByteTrack()
        self.tracker.minimum_consecutive_frames = tracker_minimum_consecutive_frames
        self.tracker.max_time_lost = max_time_lost
        self.tracker.track_activation_threshold = 0.2

        print(f"Minimum consecutive frames: {self.tracker.minimum_consecutive_frames}")
        print(f"Minimum matching threshold: {self.tracker.minimum_matching_threshold}")
        print(f"Max time lost: {self.tracker.max_time_lost}")
        print(f"Track activation threshold: {self.tracker.track_activation_threshold}")


        color = sv.ColorPalette.from_hex(['#a351fb', '#4cfb12', "#ff4040"])
        self.box_annotator = sv.BoxAnnotator(thickness=2, color=color)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.2, text_thickness=1, color=color)

        self.frame_idx = 0
        self.counted_ids = defaultdict(set)

    def _filter_detections(self, xyxy, cls, conf) -> sv.Detections:
        keep = []
        for i, (c, s) in enumerate(zip(cls, conf)):
            class_id = int(c)
            thr = self.det_conf_by_id[class_id]  
            if s >= thr:
                keep.append(i)
        if not keep:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=xyxy[keep],
            class_id=cls[keep],
            confidence=conf[keep],
        )

    def _update_counts(self, tracks: sv.Detections):
        for cid, tid in zip(tracks.class_id, tracks.tracker_id):
            cid, tid = int(cid), int(tid)
            self.counted_ids[self.id2name[cid]].add(tid)

    def _render_totals(self, frame):
        y = 40
        for cid in sorted(self.classes):
            name = self.id2name[cid]
            total = len(self.counted_ids.get(name, set()))
            cv2.putText(
                frame, f"{name}: {total}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
            )
            y += 32

    @torch.no_grad()
    def infer_bgr(self, frame_bgr):
        inputs = self.processor(images=frame_bgr, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=self.target_sizes
        )[0]

        xyxy = results["boxes"].cpu().numpy()
        cls = results["labels"].cpu().numpy()
        conf = results["scores"].cpu().numpy()

        return xyxy, cls, conf

    def process_videos(self, output_dir: str) -> list[str]:
        os.makedirs(output_dir, exist_ok=True)
        outputs = []

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), f"No se pudo abrir: {video_path}"

            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            out_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_tracked.mp4"
            )
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            self.frame_idx = 0
            self.counted_ids.clear()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_idx += 1

                if self.frame_idx == 1:
                    self.target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)

                start_time = time.time()

                xyxy, cls, conf = self.infer_bgr(frame)
                dets = self._filter_detections(xyxy, cls, conf)
                tracks = self.tracker.update_with_detections(dets)
                self._update_counts(tracks)

                elapsed_ms = (time.time() - start_time) * 1000

                annotated = self.box_annotator.annotate(scene=frame, detections=tracks)
                labels = [f"{self.id2name[int(c)]} #{int(tid)}"
                          for c, tid in zip(tracks.class_id, tracks.tracker_id)]
                annotated = self.label_annotator.annotate(scene=annotated, detections=tracks, labels=labels)
                self._render_totals(annotated)
                writer.write(annotated)

                if self.frame_idx % 30 == 0:
                    print(f"[DBG] f={self.frame_idx:05d} dets={len(dets)} "
                          f"first_conf={dets.confidence[0] if len(dets) else None} "
                          f"tracks={len(tracks)} time={elapsed_ms:.1f} ms")

            cap.release()
            writer.release()
            outputs.append(out_path)

        return outputs
