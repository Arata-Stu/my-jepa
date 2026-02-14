import os
import torch
from torchvision import datasets, tv_tensors

def prepare_batch(image, bboxes, masks, labels):
    # image: PIL or Tensor
    # bboxes: [[xmin, ymin, xmax, ymax], ...]
    # masks: [N, H, W] の 0/1 テンソル
    
    h, w = image.shape[-2:] if isinstance(image, torch.Tensor) else image.size[::-1]
    
    image = tv_tensors.Image(image)
    bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(h, w))
    masks = tv_tensors.Mask(masks)
    
    return image, {"boxes": bboxes, "masks": masks, "labels": labels}

class MultiViewCocoDataset(datasets.CocoDetection):
    """
    COCO 2017データセット専用クラス。
    rootを指定するだけで、自動的に images/annotations のパスを構成します。
    """
    def __init__(self, root, split="train", transforms=None, num_crops=2):
        image_dir = os.path.join(root, "images", f"{split}2017")
        ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")
        
        # パス存在確認
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"アノテーションファイルが見つかりません: {ann_file}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {image_dir}")

        super().__init__(image_dir, ann_file, transform=None)
        
        self.view_transforms = transforms
        self.num_crops = num_crops

    def __getitem__(self, idx):
        # 親クラスからRAWデータ（PIL画像とアノテーションリスト）を取得
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # bboxを [xmin, ymin, xmax, ymax] に変換
        boxes = [obj["bbox"] for obj in target]
        labels = [obj["category_id"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
        # COCO [x, y, w, h] -> [xmin, ymin, xmax, ymax]
        if boxes.shape[0] > 0:
            boxes[:, 2:] += boxes[:, :2]
        
        w, h = img.size
        
        # ベースとなるtv_tensors辞書の作成
        base_target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id])
        }

        # セグメンテーションマスクの処理
        if len(target) > 0 and "segmentation" in target[0]:
            masks = [self.coco.annToMask(obj) for obj in target]
            base_target["masks"] = tv_tensors.Mask(torch.stack([torch.as_tensor(m) for m in masks]))

        # 指定された数だけビューを生成
        views_img, views_target = [], []
        if self.view_transforms is not None:
            for _ in range(self.num_crops):
                # v2.Composeにより、毎回異なるランダム変換が適用される
                v_img, v_target = self.view_transforms(img, base_target)
                views_img.append(v_img)
                views_target.append(v_target)
        else:
            views_img = [img] * self.num_crops
            views_target = [base_target] * self.num_crops

        if self.num_crops == 1:
            return views_img[0], views_target[0]

        return views_img, views_target

def collate_fn(batch):
    # 最初の要素を確認して、Multi-viewかSingle-viewか判定
    is_multi_view = isinstance(batch[0][0], list)

    if not is_multi_view:
        # 通常の物体検出用のバッチ処理 (Val/Test用)
        return tuple(zip(*batch))

    # SSL用のMulti-viewバッチ処理
    imgs_list, targets_list = zip(*batch) 
    num_crops = len(imgs_list[0])
    batched_imgs, batched_targets = [], []
    
    for i in range(num_crops):
        v_imgs = [sample[i] for sample in imgs_list]
        v_targets = [sample[i] for sample in targets_list]
        batched_imgs.append(torch.stack(v_imgs))
        batched_targets.append(v_targets)
        
    return batched_imgs, batched_targets