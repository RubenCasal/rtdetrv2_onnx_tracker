#!/usr/bin/env python3
import os
from bytetrack_class import ByteTrackRTDETRv2Stable

CHECKPOINT = "./ptfue_dataset_pequeño_checkpoint/checkpoint-5000"
#CHECKPOINT = "./rtdetr-v2-r50-cppe5-finetune-2/checkpoint-500" # Modelo 480 x 480
VIDEOS = [
    "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/beach_video_slowed.mp4",
    "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/isla_de_lobos_barcos_personas.mp4",
    "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/coches_fuerte_ventura.mp4",
    
]
OUT = "./output_tracking"

CLASS_ID_TO_NAME = {0: "person", 1: "car", 2: "boat"}
CLASSES = [0, 1, 2]

DET_CONF_BY_ID = {0: 0.25, 1: 0.25, 2: 0.85}


MINIMUM_CONSECUTIVE_FRAMES = 6
MAX_TIME_LOST = 90  #frames

def main():
    os.makedirs(OUT, exist_ok=True)
    tracker = ByteTrackRTDETRv2Stable(
        ckpt_path=CHECKPOINT,
        video_paths=VIDEOS,
        classes_of_interest_ids=CLASSES,
        class_id_to_name=CLASS_ID_TO_NAME,
        det_conf_by_id=DET_CONF_BY_ID,
        tracker_minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
        max_time_lost=MAX_TIME_LOST

    )
    out_paths = tracker.process_videos(OUT)
    print("Guardados en:")
    for p in out_paths:
        print(" -", p)

if __name__ == "__main__":
    main()
