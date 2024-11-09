## Object detection using yolo and sam2


## Overview
This project combines YOLO (You Only Look Once) and SAM2 (Segment Anything Model 2) to create a powerful video object detection and segmentation pipeline. The system uses YOLO's object detection and tracking capabilities to identify objects, then leverages SAM2 for precise instance segmentation.

## Features
- Object detection and tracking using YOLO
- Instance segmentation using SAM2
- Support for multiple object instances
- Video processing capabilities
- Masked video output generation
- Consistent object tracking across frames

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Google Colab (optional, for notebook execution)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

2. Install required dependencies:
```bash
pip install ultralytics
pip install supervision
pip install opencv-python
pip install torch torchvision
```

3. Download SAM2 checkpoints:
```bash
cd checkpoints
chmod +x ./download_ckpts.sh
./download_ckpts.sh
```

## Project Structure
```
├── checkpoints/
│   └── sam2.1_hiera_tiny.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_t.yaml
├── custom_dataset/
│   ├── images/
│   ├── masked_video/
│   └── video/
└── yolo+sam2.py
```

## Usage

1. **Basic Usage**
```python
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

# Initialize SAM2
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Initialize YOLO
yolo = YOLO("yolov8s.pt")
```

2. **Process Video**
```python
# Extract frames from video
SOURCE_VIDEO = "path/to/your/video.mp4"
frames_generator = sv.get_video_frames_generator(SOURCE_VIDEO)

# Run detection and segmentation
detections = extract_detection_info(frame_folder, "yolov8s.pt")
```

3. **Generate Masked Output**
```python
with sv.VideoSink(TARGET_VIDEO.as_posix(), video_info=video_info) as sink:
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # Process frames and generate masked output
        # See full code for implementation details
```

## Key Components

### YOLO Detection
- Uses YOLO for initial object detection
- Implements object tracking for consistent instance identification
- Provides confidence scores and class predictions

### SAM2 Segmentation
- Takes YOLO bounding boxes as input
- Generates precise segmentation masks
- Maintains object identity across frames

### Video Processing
- Supports various video formats
- Generates frame-by-frame analysis
- Creates annotated output video with masks

## Configuration

### YOLO Model Selection
You can choose different YOLO models based on your needs:
- yolov8n.pt (nano) - Fastest
- yolov8s.pt (small) - Balanced
- yolov8m.pt (medium) - More accurate
- yolov8l.pt (large) - Most accurate

### SAM2 Configuration
Default configuration uses the tiny model. For different needs:
- Modify model_cfg path for different architectures
- Adjust checkpoint selection based on performance requirements

## Common Issues and Solutions

1. **Multiple Instances of Same Class**
   - Solution: Uses tracking IDs instead of class IDs
   - Ensures proper instance differentiation

2. **Memory Management**
   - Solution: Sequential frame processing
   - Batch size adjustment for different GPU capabilities

3. **Performance Optimization**
   - Use appropriate YOLO model size
   - Adjust frame processing resolution
   - Consider using SAM2 tiny model for faster processing

## Performance Tips

1. **GPU Optimization**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

2. **Frame Scaling**
```python
SCALE_FACTOR = 0.5  # Adjust based on needs
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments
- SAM2 by Meta AI Research
- Ultralytics for YOLO
- Supervision library for video processing

## Contact
For questions and feedback, please open an issue in the repository.

