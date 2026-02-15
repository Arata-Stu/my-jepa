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
    def __init__(self, root, split="train", transforms=None, num_crops=2, use_labels=True):
        image_dir = os.path.join(root, "images", f"{split}2017")
        ann_file = os.path.join(root, "annotations", f"instances_{split}2017.json")
        
        self.use_labels = use_labels
        
        super().__init__(image_dir, ann_file, transform=None)
        
        self.view_transforms = transforms
        self.num_crops = num_crops

    def __getitem__(self, idx):

        img = self._load_image(self.ids[idx])
        w, h = img.size
        
        target = {"image_id": torch.tensor([self.ids[idx]])}

        # 2. ラベルが必要な場合のみアノテーションを処理
        if self.use_labels:
            ann_ids = self.coco.getAnnIds(imgIds=self.ids[idx])
            anns = self.coco.loadAnns(ann_ids)
            
            boxes = [obj["bbox"] for obj in anns]
            labels = [obj["category_id"] for obj in anns]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            
            if boxes.shape[0] > 0:
                boxes[:, 2:] += boxes[:, :2] # XYWH -> XYXY
            
            target.update({
                "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            })

            if len(anns) > 0 and "segmentation" in anns[0]:
                masks = [self.coco.annToMask(obj) for obj in anns]
                target["masks"] = tv_tensors.Mask(torch.stack([torch.as_tensor(m) for m in masks]))

        # 3. ビューの生成
        views_img, views_target = [], []
        for _ in range(self.num_crops):
            if self.view_transforms is not None:
                # v2.Compose は target が空（image_idのみ）でも正しく動作します
                v_img, v_target = self.view_transforms(img, target)
                views_img.append(v_img)
                views_target.append(v_target)
            else:
                views_img.append(img)
                views_target.append(target)

        return (views_img[0], views_target[0]) if self.num_crops == 1 else (views_img, views_target)

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