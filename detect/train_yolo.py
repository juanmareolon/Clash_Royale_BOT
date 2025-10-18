from ultralytics import YOLO
from pathlib import Path

data_yaml = Path('BOT_CLASH_ROYALE/assets/data.yaml').resolve()
model = YOLO('yolov8s.pt') 

model.train(
    data=str(data_yaml),
    imgsz=640,
    epochs=60,                
    batch='auto',            
    device='auto',          
    workers=4,                
    amp=True,                


    cache='ram',               
    pin_memory=True,
    persistent_workers=True,

    optimizer='auto',
    lr0=0.01, lrf=0.01,
    momentum=0.937, weight_decay=5e-4,
    cos_lr=True, warmup_epochs=3,

    patience=20,
    save=True, save_period=5,
    project='runs_cr', name='cr_cards_yv8s_v1',

    degrees=5, translate=0.08, scale=0.4, shear=0.0,
    hsv_h=0.015, hsv_s=0.45, hsv_v=0.35,
    mosaic=0.8, close_mosaic=10,
    mixup=0.0, fliplr=0.3, flipud=0.0,

    seed=42
)
