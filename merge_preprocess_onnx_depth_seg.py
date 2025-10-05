import argparse
from pathlib import Path
from typing import List

DETECTION_MODEL_CHOICES = (
    "deimv2_dinov3_x_wholebody34_340query.onnx",
    "deimv2_dinov3_x_wholebody34_680query.onnx",
    "deimv2_dinov3_x_wholebody34_1750query.onnx",
)

import os
import numpy as np
import torch
import yaml
import onnx
import shutil
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine
from soa4onnx import outputs_add
from snd4onnx import remove
from sne4onnx import extraction
from sio4onnx import io_change

class Pre_model(torch.nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
    ):
        super(Pre_model, self).__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(input=x, size=(self.h, self.w))
        return x

class Post_model(torch.nn.Module):
    def __init__(
        self,
        input_h: int,
        input_w: int,
    ):
        super(Post_model, self).__init__()
        self.input_h = input_h
        self.input_w = input_w

    def forward(self, x: torch.Tensor, input_image_bgr: torch.Tensor):
        n, c, h, w = input_image_bgr.shape
        x = torch.nn.functional.interpolate(input=x, size=(h, w))
        return x

class DepthBBoxProcessor(torch.nn.Module):
    def __init__(self):
        super(DepthBBoxProcessor, self).__init__()

    def forward(self, bboxes: torch.Tensor, depth_map: torch.Tensor):
        """
        Args:
            bboxes (torch.Tensor): Tensor of shape [instances, 7] containing [batchid, classid, score, x1, y1, x2, y2].
            depth_map (torch.Tensor): Tensor of shape [batch, 1, Height, Width] representing pixel-wise depth.

        Returns:
            torch.Tensor: Tensor of shape [instances, 8] containing [batchid, classid, score, x1, y1, x2, y2, depth].
        """
        batch_ids = bboxes[:, 0].long()  # Extract batch indices
        depth_map = depth_map.squeeze(1)  # Shape: [batch, Height, Width]
        height, width = depth_map.shape[1:]

        # Convert normalized coordinates to absolute pixel values
        x1 = (bboxes[:, 3] * width).long()
        y1 = (bboxes[:, 4] * height).long()
        x2 = (bboxes[:, 5] * width).long()
        y2 = (bboxes[:, 6] * height).long()

        cx = ((x1 + x2) // 2).clamp(0, width - 1)
        cy = ((y1 + y2) // 2).clamp(0, height - 1)

        depth_values = depth_map[batch_ids, cy, cx]
        updated_bboxes = torch.cat((bboxes, depth_values.unsqueeze(1)), dim=1)

        return updated_bboxes

class BBoxProcessor(torch.nn.Module):
    def __init__(
        self,
        score_threshold: float = 0.35,
        target_class_id: float = 0.0,
    ) -> None:
        super(BBoxProcessor, self).__init__()
        self.score_threshold = score_threshold
        self.target_class_id = target_class_id

    def forward(self, detections: torch.Tensor) -> torch.Tensor:
        torch._assert(detections.dim() == 3, "detections must be [batch, rois, 6].")
        torch._assert(detections.size(0) == 1, "BBoxProcessor supports batch size == 1.")
        torch._assert(detections.size(2) == 6, "detections must be [batch, rois, 6].")

        detections_2d = detections.squeeze(0)
        class_ids = detections_2d[:, 0]
        scores = detections_2d[:, 5]
        valid_mask = (class_ids == self.target_class_id) & (scores >= self.score_threshold)
        valid_mask_full_class = scores >= self.score_threshold

        def _extract(mask: torch.Tensor, keep_score: bool) -> torch.Tensor:
            indices = torch.nonzero(mask, as_tuple=False)
            if indices.numel() == 0:
                feature_dim = 6 if keep_score else 5
                return detections.new_zeros((1, 0, feature_dim))

            indices = indices.squeeze(1)
            selected = torch.index_select(detections_2d, 0, indices)
            if keep_score:
                rois = selected
            else:
                rois_xyxy = selected[:, 1:5]
                class_stub = torch.zeros((selected.size(0), 1), dtype=selected.dtype, device=selected.device)
                rois = torch.cat((class_stub, rois_xyxy), dim=1)
            return rois

        rois = _extract(valid_mask, keep_score=False)
        rois_full = _extract(valid_mask_full_class, keep_score=True)

        return rois, rois_full

def export_bbox_processor(
    output_path: Path,
    num_rois: int,
    score_threshold: float = 0.35,
    target_class_id: float = 0.0,
) -> None:
    if num_rois < 1:
        raise ValueError("num_rois must be at least 1.")

    processor = BBoxProcessor(
        score_threshold=score_threshold,
        target_class_id=target_class_id,
    )
    processor.eval()

    dummy_detections = torch.zeros((1, num_rois, 6), dtype=torch.float32)
    dummy_detections[0, 0, 0] = target_class_id
    dummy_detections[0, 0, 5] = max(score_threshold + 1.0e-3, 1.0)
    torch.onnx.export(
        processor,
        args=(dummy_detections,),
        f=str(output_path),
        opset_version=17,
        input_names=['bbox_processor_input'],
        output_names=[
            'bbox_processor_output_bboxes',
            'bbox_processor_output_bboxes_full',
        ],
        dynamic_axes={
            'bbox_processor_output_bboxes': {0: 'num_valid_rois_body', 1: '5'},
            'bbox_processor_output_bboxes_full': {0: 'num_valid_rois_all', 1: '6'},
        },
    )

    model = onnx.load(str(output_path))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, str(output_path))

def get_num_rois_from_detection_model(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except onnx.shape_inference.InferenceError:
        pass

    if not model.graph.output:
        raise ValueError(f"No outputs found in ONNX graph: {model_path}")

    dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(dims) < 2:
        raise ValueError(f"Output tensor for {model_path} has insufficient dimensions.")

    rois_dim = dims[1]
    if rois_dim.HasField('dim_value') and rois_dim.dim_value > 0:
        return rois_dim.dim_value

    raise ValueError(f"Unable to determine fixed ROI count from {model_path} output shape.")


def reorder_onnx_outputs(model_path: Path, desired_order: List[str]) -> None:
    model = onnx.load(str(model_path))
    outputs = list(model.graph.output)
    lookup = {value.name: value for value in outputs}

    new_outputs = []
    for name in desired_order:
        if name not in lookup:
            raise ValueError(f"Output '{name}' not found in {model_path}.")
        new_outputs.append(lookup[name])

    if len(new_outputs) != len(outputs):
        raise ValueError("Desired output order does not cover all existing outputs.")

    del model.graph.output[:]
    model.graph.output.extend(new_outputs)
    onnx.save(model, str(model_path))


def main(detection_model_path: Path):
    detection_model_path = Path(detection_model_path)
    if not detection_model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {detection_model_path}")

    # dpa_H = 490 # 480->490, Multiples of 14
    # dpa_W = 644 # 640->644, Multiples of 14
    # onnx_file = f"depth_anything_v2_small_{dpa_H}x{dpa_W}.onnx"
    # shutil.copy('depth_anything_v2_small.onnx', onnx_file)
    # metric_inout = ""
    # extraction_op_name = "depthanything/Relu_output_0"

    dpa_H = 644 # 480->490, Multiples of 14
    dpa_W = 644 # 640->644, Multiples of 14
    onnx_file = f"depth_anything_v2_small_{dpa_H}x{dpa_W}.onnx"
    shutil.copy('depth_anything_v2_small.onnx', onnx_file)
    metric_inout = ""
    extraction_op_name = "depthanything/Relu_output_0"
    output_onnx_file = onnx_file

    rename(
        old_new=["/", "depthanything/"],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["1079", "depthanything/1079"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["pretrained.", "depthanything/pretrained."],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["10", "depthanything/10"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["depth_head.", "depthanything/depth_head."],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )

    extraction(
        input_op_names=['pixel_values'],
        output_op_names=[extraction_op_name],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
    )

    model_onnx = onnx.load(output_onnx_file)
    model_simp, check = simplify(
        model=model_onnx,
        overwrite_input_shapes={"pixel_values": [1,3,dpa_H,dpa_W]},
    )
    onnx.save(model_simp, output_onnx_file)

    remove(
        remove_node_names=['depthanything/depth_head/output_conv2/output_conv2.3/Relu'],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
    )

    rename(
        old_new=[extraction_op_name, "depth"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="outputs",
        search_mode="prefix_match",
    )

    ############### pre-process
    yolo_H=640
    yolo_W=640

    pre_onnx_file = f"preprocess_{yolo_H}x{yolo_W}_{dpa_H}x{dpa_W}.onnx"
    pre_model = Pre_model(h=dpa_H, w=dpa_W)
    x = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        pre_model,
        args=(x),
        f=pre_onnx_file,
        opset_version=17,
        input_names=['input_pre'],
        output_names=['output_pre'],
    )
    model_onnx1 = onnx.load(pre_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, pre_onnx_file)
    onnx_graph = rename(
        old_new=["/", "depthanything/pre/"],
        input_onnx_file_path=pre_onnx_file,
        output_onnx_file_path=pre_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    model_onnx2 = onnx.load(pre_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, pre_onnx_file)

    ############### post-process - depth
    post_onnx_file = f"postprocess_{yolo_H}x{yolo_W}_{dpa_H}x{dpa_W}.onnx"
    post_model = Post_model(input_h=yolo_H, input_w=yolo_W)
    x = torch.randn(1, 1, dpa_H, dpa_W).cpu()
    y = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        post_model,
        args=(x, y),
        f=post_onnx_file,
        opset_version=17,
        input_names=['input_post', 'input_image_bgr'],
        output_names=['depth'],
        dynamic_axes={
            'input_image_bgr' : {2: 'H', 3: 'W'},
            'depth' : {0: '1', 1: '1', 2: 'H', 3: 'W'},
        }
    )
    model_onnx1 = onnx.load(post_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, post_onnx_file)
    rename(
        old_new=["/", "depthanything/post/"],
        input_onnx_file_path=post_onnx_file,
        output_onnx_file_path=post_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "depthanything/onnx::"],
        input_onnx_file_path=post_onnx_file,
        output_onnx_file_path=post_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    model_onnx2 = onnx.load(post_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, post_onnx_file)

    ############### post-process - seg
    post_seg_onnx_file = f"postprocess_seg_{yolo_H}x{yolo_W}_{yolo_H}x{yolo_W}.onnx"
    post_model = Post_model(input_h=yolo_H, input_w=yolo_W)
    x = torch.randn(1, 1, yolo_H, yolo_W).cpu()
    y = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        post_model,
        args=(x, y),
        f=post_seg_onnx_file,
        opset_version=17,
        input_names=['input_post_seg', 'input_image_bgr'],
        output_names=['binary_masks'],
        dynamic_axes={
            'input_image_bgr' : {2: 'H', 3: 'W'},
            'binary_masks' : {0: '1', 1: '1', 2: 'H', 3: 'W'},
        }
    )
    model_onnx1 = onnx.load(post_seg_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, post_seg_onnx_file)
    rename(
        old_new=["/", "seg/post/"],
        input_onnx_file_path=post_seg_onnx_file,
        output_onnx_file_path=post_seg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "seg/post/onnx::"],
        input_onnx_file_path=post_seg_onnx_file,
        output_onnx_file_path=post_seg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    extraction(
        input_op_names=['input_post_seg', 'seg/post/Concat_1_output_0'],
        output_op_names=['binary_masks'],
        input_onnx_file_path=post_seg_onnx_file,
        output_onnx_file_path=post_seg_onnx_file,
    )
    model_onnx2 = onnx.load(post_seg_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, post_seg_onnx_file)

    ############### BBoxProcessor
    bbox_processor_onnx_file = Path("bboxes_processor.onnx")
    num_rois = get_num_rois_from_detection_model(detection_model_path)
    export_bbox_processor(
        output_path=bbox_processor_onnx_file,
        score_threshold=0.15,
        num_rois=num_rois,
    )
    model_onnx = onnx.load(str(bbox_processor_onnx_file))
    model_simp, check = simplify(model_onnx)
    onnx.save(model_simp, str(bbox_processor_onnx_file))
    rename(
        old_new=["/", "bbox_processor/"],
        input_onnx_file_path=bbox_processor_onnx_file,
        output_onnx_file_path=bbox_processor_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )

    ############### Rename DEIMv2 norm
    object_detection_file = str(detection_model_path)
    object_detection_file_wo_ext = os.path.splitext(os.path.basename(object_detection_file))[0]
    object_detection_file_w_prep = f'{object_detection_file_wo_ext}_with_prep.onnx'
    rename(
        old_new=["output_prep", "prep_div"],
        input_onnx_file_path=detection_model_path,
        output_onnx_file_path=object_detection_file_w_prep,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["prep/Mul_output_0", "output_prep"],
        input_onnx_file_path=object_detection_file_w_prep,
        output_onnx_file_path=object_detection_file_w_prep,
        mode="full",
        search_mode="prefix_match",
    )

    ############### DEIMv2 + DepthAnything
    combine(
        srcop_destop = [
            ['output_prep', 'input_pre']
        ],
        input_onnx_file_paths = [
            object_detection_file_w_prep,
            pre_onnx_file,
        ],
        output_onnx_file_path = object_detection_file_w_prep,
    )
    object_detection_file_w_depth = f'{object_detection_file_wo_ext}_with_depth.onnx'
    combine(
        srcop_destop = [
            ['output_pre', 'pixel_values']
        ],
        input_onnx_file_paths = [
            object_detection_file_w_prep,
            onnx_file,
        ],
        output_onnx_file_path = object_detection_file_w_depth,
    )
    rename(
        old_new=["depth", "deim_depth"],
        input_onnx_file_path=object_detection_file_w_depth,
        output_onnx_file_path=object_detection_file_w_depth,
        mode="outputs",
        search_mode="prefix_match",
    )
    object_detection_file_w_depth_post = f'{object_detection_file_wo_ext}_with_depth_post.onnx'
    combine(
        srcop_destop = [
            ['deim_depth', 'input_post', 'input_bgr', 'input_image_bgr'],
        ],
        input_onnx_file_paths = [
            object_detection_file_w_depth,
            post_onnx_file,
        ],
        output_onnx_file_path = object_detection_file_w_depth_post,
    )

    ############### DEIMv2/DepthAnything + BBoxPostProcessor
    combine(
        srcop_destop = [
            ['label_xyxy_score', 'bbox_processor_input'],
        ],
        input_onnx_file_paths = [
            object_detection_file_w_depth_post,
            bbox_processor_onnx_file,
        ],
        output_onnx_file_path = object_detection_file_w_depth_post,
    )
    rename(
        old_new=["bbox_processor_output_bboxes_full", "bbox_classid_xyxy_score"],
        input_onnx_file_path=object_detection_file_w_depth_post,
        output_onnx_file_path=object_detection_file_w_depth_post,
        mode="full",
        search_mode="prefix_match",
    )

    ############### DEIMv2 + InstanceSeg
    insseg_onnx_file = "best_model_b1_640x640_80x60_0.8551_dil1.onnx"
    rename(
        old_new=["model.", "insseg/model."],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["/", "insseg/"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["onnx::", "insseg/onnx::"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Gather_", "insseg/Gather_"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Range_", "insseg/Range_"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Reshape_", "insseg/Reshape_"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["1732", "insseg/1732"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["1738", "insseg/1738"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["1740", "insseg/1740"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=[" masks", "pre_masks"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["binary_masks", "pre_binary_masks"],
        input_onnx_file_path=insseg_onnx_file,
        output_onnx_file_path=insseg_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )

    full_model_file = "deimv2_depthanythingv2_instanceseg_1x3xHxW.onnx"
    combine(
        srcop_destop = [
            ['output_prep', 'images', 'bbox_processor_output_bboxes', 'rois']
        ],
        input_onnx_file_paths = [
            object_detection_file_w_depth_post,
            insseg_onnx_file,
        ],
        output_onnx_file_path = full_model_file,
    )
    # outputs_add(
    #     input_onnx_file_path=full_model_file,
    #     output_op_names=["bbox_classid_xyxy_score"],
    #     output_onnx_file_path=full_model_file,
    # )
    combine(
        srcop_destop = [
            ['pre_binary_masks', 'input_post_seg', 'depthanything/post/Concat_1_output_0', 'seg/post/Concat_1_output_0']
        ],
        input_onnx_file_paths = [
            full_model_file,
            f"postprocess_seg_{yolo_H}x{yolo_W}_{yolo_H}x{yolo_W}.onnx"
        ],
        output_onnx_file_path = full_model_file,
    )
    # rename(
    #     old_new=["masks", "instance_masks"],
    #     input_onnx_file_path=full_model_file,
    #     output_onnx_file_path=full_model_file,
    #     mode="full",
    #     search_mode="prefix_match",
    # )
    io_change(
        input_onnx_file_path=full_model_file,
        output_onnx_file_path=full_model_file,
        input_names=[
            "input_bgr",
        ],
        input_shapes=[
            [1, 3, "H", "W"],
        ],
        output_names=[
            "depth",
            "instance_masks",
            "bbox_classid_xyxy_score",
            "binary_masks",
        ],
        output_shapes=[
            [1, 1, "H", "W"],
            ["num_rois", 1, 160, 120],
            ["num_rois", 6],
            [1, 1, "H", "W"],
        ],
    )

    reorder_onnx_outputs(
        model_path=full_model_file,
        desired_order=[
            "bbox_classid_xyxy_score",
            "depth",
            "binary_masks",
            "instance_masks",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate preprocessing/postprocessing ONNX graphs and bbox processors."
    )
    parser.add_argument(
        "--detection_model",
        choices=DETECTION_MODEL_CHOICES,
        metavar="MODEL",
        required=True,
        help="Detection model ONNX file to derive ROI counts from.",
    )

    args = parser.parse_args()
    main(detection_model_path=Path(args.detection_model))
