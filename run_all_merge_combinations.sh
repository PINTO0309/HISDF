#!/usr/bin/env bash
set -euo pipefail

# Run merge_preprocess_onnx_depth_seg.py for every detection/instance segmentation model pairing.
detection_models=(
  "deimv2_hgnetv2_pico_wholebody34_340query_640x640.onnx"
  "deimv2_hgnetv2_n_wholebody34_680query_640x640.onnx"
  "deimv2_dinov3_s_wholebody34_1750query_640x640.onnx"
  "deimv2_dinov3_x_wholebody34_340query_640x640.onnx"
  "deimv2_dinov3_x_wholebody34_680query_640x640.onnx"
  "deimv2_dinov3_x_wholebody34_1750query_640x640.onnx"
)

instanceseg_models=(
  "best_model_b0_640x640_32x24_0.8416_dil1.onnx"
  "best_model_b0_640x640_64x48_0.8545_dil1.onnx"
  "best_model_b0_640x640_80x60_0.8548_dil1.onnx"
  "best_model_b0_640x640_96x72_0.8511_dil1.onnx"
  "best_model_b0_640x640_112x84_0.8495_dil1.onnx"
  "best_model_b0_640x640_128x96_0.8503_dil1.onnx"
  "best_model_b1_640x640_32x24_0.8445_dil1.onnx"
  "best_model_b1_640x640_64x48_0.8558_dil1.onnx"
  "best_model_b1_640x640_80x60_0.8551_dil1.onnx"
  "best_model_b1_640x640_96x72_0.8525_dil1.onnx"
  "best_model_b1_640x640_112x84_0.8526_dil1.onnx"
  "best_model_b1_640x640_128x96_0.8497_dil1.onnx"
  "best_model_b7_640x640_64x48_0.8547_dil1.onnx"
  "best_model_b7_640x640_80x60_0.8535_dil1.onnx"
)

for detection_model in "${detection_models[@]}"; do
  for instanceseg_model in "${instanceseg_models[@]}"; do
    echo "Running detection=${detection_model} instanceseg=${instanceseg_model}"
    uv run python merge_preprocess_onnx_depth_seg.py \
      --detection_model "${detection_model}" \
      --instanceseg_model "${instanceseg_model}"
  done
done
