import torch
import cv2
import numpy as np
from PIL import Image
from utils.logger import logger
from scipy.spatial.transform import Rotation as R
import os
import re
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor


class SpatialReasoningEnhancer:
    def __init__(self, rec_model_path, device="cuda", vl_model_path=None,
                 image_width=512, image_height=512, fov_degrees=90,
                 max_valid_depth=80, min_valid_depth=0.5,
                 sam_checkpoint_path="/project/CityNavAgent/external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"):
        self.device = device
        self.max_valid_depth = max_valid_depth
        self.min_valid_depth = min_valid_depth
        self.intrinsic_params = self._initialize_camera_intrinsics(
            image_width=image_width,
            image_height=image_height,
            fov_degrees=fov_degrees)
        self.depth_scale = 155
        config_path = "/project/CityNavAgent/external/Grounded_Sam_Lite/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "/project/CityNavAgent/external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"

        logger.info(f"Loading GroundingDINO config: {config_path}")
        args = SLConfig.fromfile(config_path)
        args.device = device
        bert_path = "/projecrt/.cache/huggingface/transformers/bert-base-uncased"
        args.bert_base_uncased_path = bert_path
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.gdino_model = model.to(device).eval()

        logger.info(f"Loading SAM model: {sam_checkpoint_path}")
        sam_model = sam_model_registry['vit_h'](checkpoint=sam_checkpoint_path)
        self.sam_predictor = SamPredictor(sam_model.to(device))

        logger.info("SAM model loading completed")
        logger.info(f"Depth filtering range: {self.min_valid_depth}-{self.max_valid_depth} meters")

    def get_detections_for_exploration_map(self, spatial_result: Dict) -> List[Dict]:
        detections = []
        for det in spatial_result.get("detections", []):
            detection = {
                "label": det.get("label", "unknown"),
                "confidence": det.get("score", det.get("confidence", 0.5)),
                "world_position": det.get("coord", [0, 0, 0]),
                "viewpoint": det.get("viewpoint", "unknown"),
                "depth": det.get("depth", 0)
            }
            detections.append(detection)
        return detections

    def _run_grounding_dino(self, image_pil, caption, box_threshold=0.3, text_threshold=0.25):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        image = image.to(self.device)
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        with torch.no_grad():
            outputs = self.gdino_model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits, boxes = logits[filt_mask], boxes[filt_mask]
        if len(boxes) == 0:
            return [], [], []
        tokenlizer = self.gdino_model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits, boxes):
            phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            score = logit.max().item()
            pred_phrases.append(phrase)
            scores.append(score)
        return boxes.numpy(), scores, pred_phrases

    def _run_sam_segmentation(self, image_rgb: np.ndarray, boxes_pixel: np.ndarray) -> List[np.ndarray]:
        if len(boxes_pixel) == 0:
            return []
        self.sam_predictor.set_image(image_rgb)
        boxes_tensor = torch.tensor(boxes_pixel, device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_tensor, image_rgb.shape[:2]
        )
        with torch.no_grad():
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
        masks_list = [mask[0].cpu().numpy() for mask in masks]
        return masks_list

    def _mask_to_world_pointcloud(
            self,
            mask: np.ndarray,
            depth_img: np.ndarray,
            camera_name: str,
            downsample_step: int = 2
    ) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.array([]).reshape(0, 3)
        indices = np.arange(0, len(xs), downsample_step)
        xs = xs[indices]
        ys = ys[indices]
        depths = depth_img[ys, xs] * self.depth_scale
        valid_mask = (depths >= self.min_valid_depth) & (depths <= self.max_valid_depth)
        xs = xs[valid_mask]
        ys = ys[valid_mask]
        depths = depths[valid_mask]
        if len(xs) == 0:
            return np.array([]).reshape(0, 3)
        fx = self.intrinsic_params['fx']
        fy = self.intrinsic_params['fy']
        cx = self.intrinsic_params['cx']
        cy = self.intrinsic_params['cy']
        x_c = (xs - cx) * depths / fx
        y_c = (ys - cy) * depths / fy
        z_c = depths
        points_body = np.stack([z_c, x_c, y_c], axis=1)
        cam = self.extrinsic_params[camera_name]
        R_c2w = cam['R_c2w']
        T_c2w = cam['T_c2w']
        points_world = (R_c2w @ points_body.T).T + T_c2w
        return points_world

    def _estimate_bounding_box(self, points: np.ndarray) -> Optional[Dict]:
        if len(points) == 0:
            return None
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        height_above_ground = -min_bound[2]
        return {
            'min_bound': min_bound.tolist(),
            'max_bound': max_bound.tolist(),
            'center': center.tolist(),
            'size': size.tolist(),
            'num_points': len(points),
            'height_above_ground': round(height_above_ground, 2)
        }

    def _initialize_camera_intrinsics(self, image_width=512, image_height=512, fov_degrees=90):
        fov_rad = np.radians(fov_degrees)
        fx = image_width / (2 * np.tan(fov_rad / 2))
        fy = image_height / (2 * np.tan(fov_rad / 2))
        cx, cy = image_width / 2, image_height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        return {'K': K, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    def _compute_camera_extrinsics(self, viewpoint_poses_dict):
        camera_params = {}
        for cam_name, pose in viewpoint_poses_dict.items():
            if len(pose) >= 7:
                pos = pose[:3]
                qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
                quat = [qx, qy, qz, qw]
                R_w2b = R.from_quat(quat).as_matrix()
                R_c2w = R_w2b
                camera_params[cam_name] = {
                    'R_c2w': R_c2w,
                    'T_c2w': np.array(pos),
                    'pose_raw': pose
                }
        return camera_params

    def transform_pixel_to_world(self, pixel_x, pixel_y, depth_val, camera_name='front'):
        if camera_name not in self.extrinsic_params:
            raise ValueError(f"Camera '{camera_name}' not registered. Available: {list(self.extrinsic_params.keys())}")
        fx = self.intrinsic_params['fx']
        fy = self.intrinsic_params['fy']
        cx = self.intrinsic_params['cx']
        cy = self.intrinsic_params['cy']
        x_c = (pixel_x - cx) * depth_val / fx
        y_c = (pixel_y - cy) * depth_val / fy
        z_c = depth_val
        point_body = np.array([z_c, x_c, y_c])
        cam = self.extrinsic_params[camera_name]
        point_world = cam['R_c2w'] @ point_body + cam['T_c2w']
        return point_world.tolist()

    def _numpy_to_base64(self, img_array: np.ndarray) -> str:
        import cv2
        import base64
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        success, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError("Image encoding failed")

    def get_visual_descriptions_and_landmarks(
            self,
            viewpoint_rgb_imgs: Dict[str, Any],
            instruction: str,
            llm_client,
            all_landmarks: List[str] = None,
            image_order: List[str] = None
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        if image_order is None:
            image_order = ["left", "slightly_left", "front", "slightly_right", "right"]
        visual_descriptions = {}
        per_view_landmarks = {}
        IRRELEVANT_OBJECTS = {
            'sun', 'sky', 'cloud', 'clouds', 'moon', 'stars', 'star',
            'shadow', 'shadows', 'light', 'sunlight', 'reflection',
            'horizon', 'background', 'foreground', 'rooftops', 'rooftop'}
        prompt_template = (
            "You are a navigation assistant helping a user complete the following navigation task.\n\n"
            "# Navigation Instruction\n"
            "\"{instruction}\"\n\n"
            "# Candidate Landmarks (MUST SELECT ONLY FROM THIS LIST)\n"
            "{landmarks_list}\n\n"
            "# Definitions of Ambiguous Landmarks\n"
            "- 'park': refers to a green urban open space with grass, trees, benches, or recreational facilities — **NOT** a road, parking lot, empty asphalt, gray ground, or any bare surface without vegetation.\n"
            "- 'end of the park': only valid if a real 'park' is visible and you can see its boundary or exit area.\n\n"
            "# Task\n"
            "Identify which candidate landmarks from the list above are **unambiguously, clearly, and confidently visible** in this image.\n\n"
            "# Output Format\n"
            "object_name:feature_description[position_in_image][distance]\n\n"
            "Where:\n"
            "- object_name: **MUST be an EXACT string from the Candidate Landmarks list above.**\n"
            "- **DO NOT output 'main road', 'intersection', or any phrase not in the list.**\n"
            "- **If NO candidate landmark is clearly visible, output: (no prominent objects)**\n"
            "- feature_description: Brief visual features (e.g., color, material, height, context)\n"
            "- position_in_image: left / center-left / center / center-right / right\n"
            "- distance: near (0-15m) / medium (15-50m) / far (>50m)\n\n"
            "# Examples (GOOD)\n"
            "white building:white facade, 4 stories, grid-patterned windows[center][near]\n"
            "black building:dark tall structure with glass panels[right][medium]\n"
            "park:green open area with trees and benches[left][medium]\n"
            "(no prominent objects)\n\n"
            "# Rules\n"
            "- **ONLY output landmark names that appear EXACTLY in the Candidate Landmarks list.**\n"
            "- **NEVER invent, abbreviate, or rephrase landmark names (e.g., do not say 'park area' or 'end of green').**\n"
            "- **DO NOT label roads, asphalt, sidewalks, or gray empty ground as 'park' or 'end of the park'.**\n"
            "- **If you see only roads, sky, or empty space, output (no prominent objects).**\n"
            "- **When in doubt, OMIT the object — conservative output is strongly preferred.**\n"
            "- Ignore: sky, clouds, sun, shadows, reflections, distant blurry structures.\n"
            "- Maximum 5 objects per image.\n"
            "- One object per line, no numbering, bullets, or extra text.\n\n"
            "Now describe this image:"
        )
        if all_landmarks and len(all_landmarks) > 0:
            landmarks_list = "\n".join([f"  - {lm}" for lm in all_landmarks])
        else:
            landmarks_list = "  (No specific landmarks - describe navigation-relevant objects)"
        for view_name in image_order:
            try:
                if isinstance(viewpoint_rgb_imgs, dict):
                    rgb_img = viewpoint_rgb_imgs.get(view_name)
                elif isinstance(viewpoint_rgb_imgs, list):
                    idx = image_order.index(view_name)
                    rgb_img = viewpoint_rgb_imgs[idx] if idx < len(viewpoint_rgb_imgs) else None
                else:
                    rgb_img = None
                if rgb_img is None:
                    logger.warning(f"  [{view_name}]: No image data")
                    visual_descriptions[view_name] = ""
                    per_view_landmarks[view_name] = []
                    continue
                prompt = prompt_template.format(
                    instruction=instruction,
                    landmarks_list=landmarks_list
                )
                base64_img = None
                try:
                    if isinstance(rgb_img, str):
                        if os.path.exists(rgb_img):
                            img_array = cv2.imread(rgb_img)
                            if img_array is not None:
                                success, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if success:
                                    base64_img = base64.b64encode(buffer).decode('utf-8')
                        else:
                            base64_img = rgb_img
                    elif isinstance(rgb_img, np.ndarray):
                        base64_img = self._numpy_to_base64(rgb_img)
                    elif isinstance(rgb_img, Image.Image):
                        buffer = BytesIO()
                        rgb_img.convert('RGB').save(buffer, format='JPEG', quality=85)
                        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
                except Exception as e:
                    logger.error(f"  [{view_name}]: Image conversion failed - {e}")
                if base64_img is None:
                    logger.warning(f"  [{view_name}]: base64 conversion failed")
                    visual_descriptions[view_name] = ""
                    per_view_landmarks[view_name] = []
                    continue
                images_dict = {view_name: base64_img}
                response = llm_client.query_image_api(
                    prompt=prompt,
                    images=images_dict,
                    show_response=False
                )
                response_text = response.strip()
                visual_descriptions[view_name] = response_text
                landmarks = []
                if "(no prominent objects)" not in response_text.lower():
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if ':' in line:
                            object_name = line.split(':')[0].strip()
                            object_name = object_name.strip('-').strip('*').strip('•').strip()
                            if object_name and len(object_name) > 1:
                                name_lower = object_name.lower()
                                is_irrelevant = any(
                                    irrelevant in name_lower
                                    for irrelevant in IRRELEVANT_OBJECTS
                                )
                                if not is_irrelevant:
                                    landmarks.append(object_name)
                per_view_landmarks[view_name] = landmarks
            except Exception as e:
                logger.warning(f"  [{view_name}]: Failed - {e}")
                visual_descriptions[view_name] = ""
                per_view_landmarks[view_name] = []
        logger.info("📋 Extracted landmarks per view:")
        for view_name, landmarks in per_view_landmarks.items():
            logger.info(f"  [{view_name}]: {landmarks}")
        return visual_descriptions, per_view_landmarks

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        def to_xyxy(box):
            cx, cy, w, h = box
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

        b1 = to_xyxy(box1)
        b2 = to_xyxy(box2)
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _apply_nms_to_detections(
            self,
            detections: List[Dict],
            iou_threshold: float = 0.5
    ) -> List[Dict]:
        if len(detections) == 0:
            return []
        sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        suppressed = [False] * len(sorted_dets)
        for i, det_i in enumerate(sorted_dets):
            if suppressed[i]:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue
                iou = self._compute_iou(
                    np.array(det_i['box']),
                    np.array(sorted_dets[j]['box'])
                )
                if iou > iou_threshold:
                    suppressed[j] = True
        return keep

    def _filter_by_semantic_height_constraints(
            self,
            label: str,
            z_coord: float
    ) -> Tuple[bool, str]:
        height_above_ground = -z_coord
        label_lower = label.lower()
        SEMANTIC_HEIGHT_CONSTRAINTS = {
            'ground': {
                'keywords': ['road', 'street', 'sidewalk', 'crosswalk', 'path',
                             'lane', 'intersection', 'pavement', 'asphalt',
                             'parking', 'ground', 'floor', 'park'],
                'min_height': -20.5,
                'max_height': 15.0
            },
            'low_structure': {
                'keywords': ['traffic light', 'sign', 'pole', 'bench', 'bus stop',
                             'streetlight', 'lamp', 'hydrant', 'mailbox', 'fence',
                             'barrier', 'bollard', 'trash', 'bin'],
                'min_height': -20.5,
                'max_height': 15.0
            },
            'vegetation': {
                'keywords': ['tree', 'bush', 'hedge', 'shrub', 'plant', 'grass',
                             'garden', 'lawn'],
                'min_height': -20.0,
                'max_height': 15.0
            },
            'vehicle': {
                'keywords': ['car', 'bus', 'truck', 'vehicle', 'taxi', 'van',
                             'motorcycle', 'bike', 'bicycle'],
                'min_height': -20.0,
                'max_height': 10.0
            },
            'building': {
                'keywords': ['building', 'tower', 'house', 'apartment', 'office',
                             'skyscraper', 'mall', 'store', 'shop', 'hotel',
                             'hospital', 'school', 'church', 'station'],
                'min_height': -20.0,
                'max_height': 80.0
            }
        }
        matched_category = None
        min_h = None
        max_h = None
        for category, constraints in SEMANTIC_HEIGHT_CONSTRAINTS.items():
            for keyword in constraints['keywords']:
                if keyword in label_lower:
                    matched_category = category
                    min_h = constraints['min_height']
                    max_h = constraints['max_height']
                    break
            if matched_category:
                break
        if matched_category is None:
            return True, ""
        if height_above_ground < min_h or height_above_ground > max_h:
            reason = (f"Semantic height unreasonable: {label} height={height_above_ground:.1f}m, "
                      f"Category={matched_category}, Reasonable range=[{min_h}, {max_h}]m")
            return False, reason
        return True, ""

    def perform_spatial_reasoning_with_mllm_landmarks(
            self,
            viewpoint_rgb_imgs: Dict,
            viewpoint_dep_imgs: Dict,
            viewpoint_poses: Dict,
            per_view_landmarks: Dict[str, List[str]],
            image_order: List[str] = None,
            box_threshold: float = 0.3,
            text_threshold: float = 0.2,
            enable_sam: bool = True,
            downsample_step: int = 2,
            save_masks: bool = True,
            nms_iou_threshold: float = 0.5
    ) -> Dict:
        if image_order is None:
            image_order = ["left", "slightly_left", "front", "slightly_right", "right"]
        self.extrinsic_params = self._compute_camera_extrinsics(viewpoint_poses)
        spatial_context = (
            "All positions are in AirSim NED world coordinates: X=forward (north), Y=right (east), Z=down. "
            "Landmark coordinates below are global 3D positions."
        )
        total_detections = 0
        all_detections = []
        depth_filter_stats = {
            'total_boxes': 0,
            'filtered_by_depth': 0,
            'filtered_by_bounds': 0,
            'filtered_by_semantic_height': 0,
            'valid_detections': 0,
            'nms_suppressed': 0
        }
        for vp in image_order:
            if not all(vp in d for d in [viewpoint_rgb_imgs, viewpoint_dep_imgs, viewpoint_poses]):
                logger.warning(f"⚠️ Skipping viewpoint {vp} (data missing)")
                continue
            landmarks_for_view = per_view_landmarks.get(vp, [])
            rgb_img = viewpoint_rgb_imgs[vp]
            depth_img = viewpoint_dep_imgs[vp].squeeze()
            if isinstance(rgb_img, np.ndarray):
                rgb_img_bgr = rgb_img
            else:
                rgb_img_bgr = np.array(rgb_img)
            rgb_img_rgb = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img_rgb)
            W, H = pil_img.size
            raw_detections_for_view = []
            for single_landmark in landmarks_for_view:
                try:
                    boxes, scores, _ = self._run_grounding_dino(
                        pil_img, single_landmark,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold
                    )
                    if len(boxes) == 0:
                        logger.debug(f"  {vp}/{single_landmark}: Not detected")
                        continue
                    for box, score in zip(boxes, scores):
                        raw_detections_for_view.append({
                            'box': box.tolist() if isinstance(box, np.ndarray) else box,
                            'score': score,
                            'label': single_landmark
                        })
                    logger.debug(f"  {vp}/{single_landmark}: Detected {len(boxes)} boxes")
                except Exception as e:
                    logger.warning(f"  {vp}/{single_landmark}: Detection failed - {e}")
                    continue
            if len(raw_detections_for_view) == 0:
                logger.debug(f"  {vp}: All landmarks not detected")
                continue
            before_nms_count = len(raw_detections_for_view)
            nms_detections = self._apply_nms_to_detections(
                raw_detections_for_view,
                iou_threshold=nms_iou_threshold
            )
            after_nms_count = len(nms_detections)
            nms_suppressed = before_nms_count - after_nms_count
            depth_filter_stats['nms_suppressed'] += nms_suppressed
            if nms_suppressed > 0:
                logger.info(
                    f"  {vp}: NMS deduplication {before_nms_count} → {after_nms_count} (suppressed {nms_suppressed})")
            boxes_pixel_list = []
            valid_nms_detections = []
            for det in nms_detections:
                box = det['box']
                cx_norm, cy_norm, w_norm, h_norm = box
                x1 = (cx_norm - w_norm / 2) * W
                y1 = (cy_norm - h_norm / 2) * H
                x2 = (cx_norm + w_norm / 2) * W
                y2 = (cy_norm + h_norm / 2) * H
                boxes_pixel_list.append([x1, y1, x2, y2])
                valid_nms_detections.append(det)
            boxes_pixel = np.array(boxes_pixel_list) if boxes_pixel_list else np.array([]).reshape(0, 4)
            masks_list = []
            if enable_sam and len(boxes_pixel) > 0:
                try:
                    masks_list = self._run_sam_segmentation(rgb_img_rgb, boxes_pixel)
                    logger.debug(f"  {vp}: SAM segmentation completed, generated {len(masks_list)} masks")
                except Exception as e:
                    logger.warning(f"  {vp}: SAM segmentation failed - {e}")
                    masks_list = [None] * len(boxes_pixel)
            else:
                masks_list = [None] * len(boxes_pixel)
            viewpoint_objects = []
            for i, det in enumerate(valid_nms_detections):
                depth_filter_stats['total_boxes'] += 1
                box = det['box']
                score = det['score']
                label = det['label']
                cx_norm, cy_norm, w_norm, h_norm = box
                x1 = (cx_norm - w_norm / 2) * W
                y1 = (cy_norm - h_norm / 2) * H
                x2 = (cx_norm + w_norm / 2) * W
                y2 = (cy_norm + h_norm / 2) * H
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if not (0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]):
                    depth_filter_stats['filtered_by_bounds'] += 1
                    continue
                depth_val = depth_img[cy, cx] * self.depth_scale
                if not (self.min_valid_depth <= depth_val <= self.max_valid_depth):
                    depth_filter_stats['filtered_by_depth'] += 1
                    logger.debug(
                        f"🚫 Filtering detection box: {label} depth {depth_val:.1f}m out of range "
                        f"[{self.min_valid_depth}, {self.max_valid_depth}]m"
                    )
                    continue
                try:
                    world_pos = self.transform_pixel_to_world(cx, cy, depth_val, vp)
                    coord_3d = [round(p, 1) for p in world_pos]
                    is_height_valid, filter_reason = self._filter_by_semantic_height_constraints(
                        label=label,
                        z_coord=coord_3d[2]
                    )
                    if not is_height_valid:
                        depth_filter_stats['filtered_by_semantic_height'] += 1
                        logger.debug(f"🚫 {filter_reason}")
                        continue
                    detection = {
                        'label': label,
                        'score': score,
                        'coord': coord_3d,
                        'depth': depth_val,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'viewpoint': vp
                    }
                    if enable_sam and i < len(masks_list) and masks_list[i] is not None:
                        mask = masks_list[i]
                        pointcloud = self._mask_to_world_pointcloud(
                            mask, depth_img, vp,
                            downsample_step=downsample_step
                        )
                        bbox_3d = self._estimate_bounding_box(pointcloud)
                        detection['pointcloud'] = pointcloud
                        detection['bbox_3d'] = bbox_3d
                        if bbox_3d:
                            detection['height_above_ground'] = bbox_3d['height_above_ground']
                        if save_masks:
                            detection['mask'] = mask
                    viewpoint_objects.append(detection)
                    all_detections.append(detection)
                    depth_filter_stats['valid_detections'] += 1
                except Exception as e:
                    logger.debug(f"⚠️ {label} coordinate transformation failed: {e}")
            if viewpoint_objects:
                def format_obj(obj):
                    label = obj['label']
                    conf = obj['score']
                    cx, cy, cz = obj['coord']
                    depth = obj['depth']
                    height_str = ""
                    if 'height_above_ground' in obj:
                        height_str = f",h={obj['height_above_ground']:.1f}m"
                    bbox_info = ""
                    if 'bbox_3d' in obj and obj['bbox_3d']:
                        size = obj['bbox_3d']['size']
                        bbox_info = f"[size={[round(s, 1) for s in size]}]"
                    return f"{label}(conf={conf:.2f})@[{cx},{cy},{cz}](depth={depth:.1f}m){bbox_info}"

                objects_str = ", ".join(format_obj(obj) for obj in viewpoint_objects)
                spatial_context += f"\n {vp}: {objects_str}"
                total_detections += len(viewpoint_objects)
                logger.info(f"✅ {vp}: Detected {len(viewpoint_objects)} objects")
                for obj in viewpoint_objects:
                    bbox_info = ""
                    if 'bbox_3d' in obj and obj['bbox_3d']:
                        bbox_info = f", bbox_size={[round(s, 1) for s in obj['bbox_3d']['size']]}"
                    logger.info(
                        f"  └─ {obj['label']} @ {obj['coord']}, "
                        f"conf={obj['score']:.2f}, depth={obj['depth']:.1f}m{bbox_info}"
                    )
        logger.info(
            f"📊 Detection stats: Total boxes {depth_filter_stats['total_boxes']} | "
            f"Valid detections {depth_filter_stats['valid_detections']} | "
            f"Depth filtered {depth_filter_stats['filtered_by_depth']} | "
            f"Boundary filtered {depth_filter_stats['filtered_by_bounds']} | "
            f"Semantic height filtered {depth_filter_stats['filtered_by_semantic_height']} | "
            f"NMS suppressed {depth_filter_stats['nms_suppressed']}"
        )
        logger.info(f"📍 Total detected {total_detections} objects")
        return {
            "text_context": spatial_context,
            "detections": all_detections,
            "depth_stats": depth_filter_stats
        }

    def perform_mllm_guided_spatial_reasoning(
            self,
            viewpoint_rgb_imgs: Dict,
            viewpoint_dep_imgs: Dict,
            viewpoint_poses: Dict,
            instruction: str,
            original_landmarks: List[str],
            all_landmarks: List[str],
            llm_client,
            image_order: List[str] = None,
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
            enable_sam: bool = True,
            downsample_step: int = 2,
            save_masks: bool = True,
            nms_iou_threshold: float = 0.5
    ) -> Dict:
        logger.info("🚀 Starting MLLM-guided spatial reasoning (per-label detection mode)...")
        if enable_sam:
            logger.info("🎭 SAM segmentation mode enabled")
        if image_order is None:
            image_order = ["left", "slightly_left", "front", "slightly_right", "right"]
        visual_descriptions, per_view_landmarks = self.get_visual_descriptions_and_landmarks(
            viewpoint_rgb_imgs=viewpoint_rgb_imgs,
            instruction=instruction,
            llm_client=llm_client,
            all_landmarks=all_landmarks,
            image_order=image_order)
        spatial_result = self.perform_spatial_reasoning_with_mllm_landmarks(
            viewpoint_rgb_imgs=viewpoint_rgb_imgs,
            viewpoint_dep_imgs=viewpoint_dep_imgs,
            viewpoint_poses=viewpoint_poses,
            per_view_landmarks=per_view_landmarks,
            image_order=image_order,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            enable_sam=enable_sam,
            downsample_step=downsample_step,
            save_masks=save_masks,
            nms_iou_threshold=nms_iou_threshold)
        if enable_sam:
            detections_with_pc = sum(
                1 for d in spatial_result['detections'] if 'pointcloud' in d and len(d['pointcloud']) > 0)
            logger.info(
                f"📊 {detections_with_pc}/{len(spatial_result['detections'])} detections contain point cloud data")
        return {
            "text_context": spatial_result["text_context"],
            "visual_descriptions": visual_descriptions,
            "per_view_landmarks": per_view_landmarks,
            "detections": spatial_result["detections"],
            "depth_stats": spatial_result["depth_stats"]}
