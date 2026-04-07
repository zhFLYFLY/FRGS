import sys
sys.path.append(".GroundingDINO")
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

import logging

logger = logging.getLogger("groundingdino-sam")
logger.setLevel(logging.INFO)

import os
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

import cv2
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


class GroundedSam:
    def __init__(
            self,
            dino_checkpoint_path="/project/CityNavAgent/external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth",
            sam_checkpoint_path="/project/CityNavAgent/external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth",
            dino_config_path="external/Grounded_Sam_Lite/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device='cuda',
    ):
        self.device = device
        self.dino = self.load_dino_model(dino_config_path, dino_checkpoint_path, bert_base_uncased_path=None, device=device)
        sam_checkpoint_path = "/project/CityNavAgent/external/Grounded_Sam_Lite/weights/sam_vit_h_4b8939.pth"
        self.sam = SamPredictor(sam_model_registry['vit_h'](checkpoint=sam_checkpoint_path).to(device))

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_dino_model(self, model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
        dino_config_path = ("/project/CityNavAgent/external/Grounded_Sam_Lite/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/Groundi"
                            "ngDINO_SwinT_OGC.py")
        args = SLConfig.fromfile(dino_config_path)
        args.device = device

        if bert_base_uncased_path is None:
            bert_base_uncased_path = "/home/cache/huggingface/transformers/bert-base-uncased"

        args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        model_checkpoint_path="/project/CityNavAgent/external/Grounded_Sam_Lite/weights/groundingdino_swint_ogc.pth"
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_dino_output(self, image, caption, box_threshold, text_threshold, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = self.dino.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=5))
        ax.text(x0, y0, label)

    def greedy_mask_predict(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25, visualize=False,
                            save_path=None):
        seg_success = True
        h, w = image.shape[:2]
        in_phrases = text_prompt.split(".")
        in_phrases = [inp.strip(" ") for inp in in_phrases]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil_transformed, _ = self.transform(image_pil, None)

        logger.info(f"调用Grounded 进行目标检测；text_prompt: '{text_prompt}'")
        pred_boxes, pred_phrases = self.get_dino_output(
            image_pil_transformed, text_prompt, box_threshold, text_threshold, with_logits=True
        )

        logger.info(f"📦 Grounding DINO 检测到 {len(pred_phrases)} 个候选框")

        pred_boxes = pred_boxes.cpu()

        if len(pred_phrases) == 0:
            seg_success = False
            return np.zeros((h, w), dtype=np.bool_), seg_success

        best_phrase = "<inf>"
        best_bboxes = []

        def _soft_contain(ele, ele_list):
            for e in ele_list:
                if ele in e:
                    return True
            return False

        for inp in in_phrases:
            if _soft_contain(inp, pred_phrases):
                best_phrase = inp
                break

        if best_phrase not in in_phrases:
            seg_success = False
            return np.zeros((h, w), dtype=np.bool_), seg_success

        for i, pp in enumerate(pred_phrases):
            if best_phrase in pp:
                best_bboxes.append(pred_boxes[i:i + 1, :])

        boxes_filt = torch.cat(best_bboxes, dim=0)

        boxes_pixel = boxes_filt.clone()
        for i in range(boxes_pixel.size(0)):
            boxes_pixel[i] = boxes_pixel[i] * torch.Tensor([w, h, w, h])
            boxes_pixel[i][:2] -= boxes_pixel[i][2:] / 2
            boxes_pixel[i][2:] += boxes_pixel[i][:2]

        self.sam.set_image(image_rgb)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_pixel, image_rgb.shape[:2]).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        masks = masks.cpu()
        final_mask = torch.any(masks[0], dim=0).numpy()


        if visualize or save_path:
            self.visualize_results(
                image_rgb,
                boxes_pixel.numpy(),
                pred_phrases,
                masks[0].numpy(),
                final_mask,
                best_phrase,
                save_path
            )

        return final_mask, seg_success

    def visualize_results(self, image_rgb, boxes, phrases, masks, final_mask, best_phrase, save_path=None):

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("Original Image", fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image_rgb)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
        for idx, (box, phrase) in enumerate(zip(boxes, phrases)):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3,
                edgecolor=colors[idx],
                facecolor='none'
            )
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(
                x1, y1 - 5,
                phrase,
                fontsize=10,
                color='white',
                bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8)
            )
        axes[0, 1].set_title(f"Detection Results (matched: '{best_phrase}')", fontsize=14)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(image_rgb)
        for idx, mask in enumerate(masks):
            color = np.concatenate([colors[idx][:3], [0.5]])
            h, w = mask.shape
            mask_overlay = np.zeros((h, w, 4))
            mask_overlay[mask] = color
            axes[1, 0].imshow(mask_overlay)
        axes[1, 0].set_title("Individual Masks Overlay", fontsize=14)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(image_rgb)
        mask_color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = final_mask.shape
        mask_overlay = np.zeros((h, w, 4))
        mask_overlay[final_mask] = mask_color
        axes[1, 1].imshow(mask_overlay)
        contours, _ = cv2.findContours(
            final_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2:
                axes[1, 1].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
        axes[1, 1].set_title("Final Merged Mask", fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到: {save_path}")

        plt.show()
        plt.close()

    def visualize_mask_comparison(self, image_rgb, final_mask, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(final_mask, cmap='gray')
        axes[1].set_title("Binary Mask", fontsize=14)
        axes[1].axis('off')

        overlay = image_rgb.copy()
        overlay[final_mask] = overlay[final_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title("Mask Overlay", fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    def predict(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25, visualize=False):
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil, _ = self.transform(image_pil, None)

        boxes_filt, pred_phrases = self.get_dino_output(image_pil, text_prompt, box_threshold, text_threshold)

        self.sam.set_image(image)
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([w, h, w, h])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    predictor = GroundedSam(dino_config_path='groundingdino/config/GroundingDINO_SwinT_OGC.py',
                            dino_checkpoint_path='weights/groundingdino_swint_ogc.pth',
                            sam_checkpoint_path='weights/sam_vit_h_4b8939.pth',
                            device='cuda')

    img = cv2.imread("assets/demo9.jpg")
    predictor.predict(img, "bear", visualize=True)
    predictor.greedy_mask_predict(img, "bear.painting on the wall.dog", visualize=True)