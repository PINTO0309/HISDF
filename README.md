# HISDF

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/HISDF)

Merging DEIMv2 and DepthAnythingV2, RHIS

HISDF (Human Instance, Skeleton, and Depth Fusion) is a unified model that fuses human instance segmentation, skeletal structure estimation, and depth prediction to achieve holistic human perception from visual input.

https://github.com/user-attachments/assets/2fcad55b-775c-46b7-bcff-1132baa7a89d

https://github.com/user-attachments/assets/6c86e438-2953-4e69-8b29-e1f77846ad03

https://github.com/user-attachments/assets/15f35c51-9aef-43dc-a128-b34d3610825e

## Features

- Multitask inference that combines person detection, attribute estimation, skeleton keypoints, and per-pixel depth from a single forward pass.
- Instance segmentation masks and keypoint overlays with stable colouring driven by a lightweight SORT-style tracker.
- Optional depth-map and mask compositing overlays to visualise the fused predictions in real time or on saved frames.
- Support for CPU, CUDA, and TensorRT execution via ONNX Runtime providers.
- Utilities for exporting detections to YOLO format and automatically persisting rendered frames or videos.

## Repository Layout

```
├── demo_hisdf_onnx_34.py        # Main demo / visualisation entry point
├── merge_preprocess_onnx_depth_seg.py
├── *.onnx                       # Trained HISDF, depth, and post-processing models
├── pyproject.toml               # Python package metadata and dependencies
├── uv.lock                      # Reproducible environment lock file
└── README.md
```

## Model Zoo

The repository ships with several ONNX artefacts:

- `deimv2_dinov3_x_wholebody34_*.onnx`: core HISDF detectors with varying query counts.
- `depth_anything_v2_small_*.onnx`: depth backbones at different input resolutions.
- `postprocess_*` / `preprocess_*`: helper networks for resizing, segmentation, or depth refinement.
- `bboxes_processor.onnx`: post-processing utilities for bounding-box outputs.

Pick the variant that best matches your latency/accuracy trade-offs. The demo defaults to `deimv2_depthanythingv2_instanceseg_1x3xHxW.onnx`.

## Installation

If you use `uv`, simply run `uv sync` to materialise the locked environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
export PYTHONWARNINGS="ignore"
```


### GPU / TensorRT notes

- CUDA provider needs the matching CUDA toolkit and cuDNN runtime.
- TensorRT execution (`--execution_provider tensorrt`) requires a built engine cache directory with write access.

## Quick Start

### Run on a webcam

```bash
python demo_hisdf_onnx_34.py \
--model deimv2_depthanythingv2_instanceseg_1x3xHxW.onnx \
--video 0 \
--execution_provider cuda
```

### Run on an image directory

```bash
python demo_hisdf_onnx_34.py \
--images_dir ./samples \
--disable_waitKey
```

Rendered images will be written to `./output` together with YOLO annotations when `--output_yolo_format_text` is enabled.

## Key Command-Line Flags

| Flag | Description |
| --- | --- |
| `--video`, `--images_dir` | Choose between streaming input and batch image inference (one is required). |
| `--execution_provider {cpu,cuda,tensorrt}` | Select the ONNX Runtime backend. |
| `--object_socre_threshold` | Detection confidence for object classes. |
| `--attribute_socre_threshold` | Confidence threshold for attribute heads. |
| `--keypoint_threshold` | Minimum score for bone/keypoint visualisation. |
| `--disable_*` toggles | Turn off attribute-specific rendering (generation, gender, handedness, head pose). |
| `--disable_video_writer` | Skip MP4 recording when reading from a video source. |
| `--enable_face_mosaic` | Pixelate face detections to preserve privacy. |
| `--output_yolo_format_text` | Export YOLO labels alongside rendered frames. |

Run `python demo_hisdf_onnx_34.py --help` for the full list.

## Visualisation Controls

While the demo window is active you can toggle features with the keyboard:

- `n` Generation (adult/child) display
- `g` Gender colouring
- `p` Head pose labelling
- `h` Handedness identification
- `k` Keypoint drawing mode (`dot` → `box` → `both`)
- `f` Face mosaic
- `b` Skeleton visibility
- `d` Depth-map overlay
- `i` Instance mask overlay
- `m` Head-distance measurement

Persistent track IDs for person boxes are drawn outside the bounding boxes. Colours are locked per track and shared with the instance masks to avoid flicker when detection ordering changes.

## Development Tips

- `demo_hisdf_onnx_34.py` is structured around a `HISDF` model wrapper that encapsulates preprocessing, inference, and postprocessing for all tasks.
- The lightweight `SimpleSortTracker` keeps body detections stable across frames; tune the IoU threshold or `max_age` if you encounter ID churn.
- Depth overlays rely on OpenCV colormaps and the segmentation mask to blend only foreground pixels.
- Use `python -m compileall demo_hisdf_onnx_34.py` to perform a quick syntax check after edits.

## Troubleshooting

- **No window appears**: Ensure an X server is available, or run headless by setting `--disable_waitKey` and writing frames to disk.
- **ONNX Runtime errors**: Confirm the selected execution provider matches your hardware and that all provider-specific dependencies are installed.
- **Incorrect colours or masks**: Make sure the instance segmentation overlay is enabled (`i` key) and verify that your model outputs masks.
- **Slow FPS**: Disable depth and mask overlays, lower the input resolution, or switch to TensorRT.

## Licensing
This project is licensed under the Apache License Version 2.0 License.

Refer to [LICENSE](./LICENSE) for the full terms governing the use of the code and bundled models.
