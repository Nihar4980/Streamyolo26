import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
box_transform = A.Compose(
    [
        A.Resize(600, 960),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["class_labels"],
        clip_bboxes_on_input=True,
        clip_after_transform=True,
        filter_invalid_bboxes=True,
    ),
)

image_transform = A.Compose(
    [
        A.Resize(600, 960),
        ToTensorV2(),
    ],
)
box_transform_flip = A.Compose(
    [
        A.Resize(600, 960),
        A.HorizontalFlip(p=1.0),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["class_labels"],
        clip_bboxes_on_input=True,
        clip_after_transform=True,
        filter_invalid_bboxes=True,
    ),
)
image_transform_flip = A.Compose(
    [
        A.Resize(600, 960),
        A.HorizontalFlip(p=1.0),
        ToTensorV2(),
    ],
)


class AgroverseDataset(Dataset):
    def __init__(self, image_path, label_path, box_transform, image_transform):
        self.image_root = Path(image_path)
        self.label_root = Path(label_path)
        self.box_transform = box_transform
        self.image_transform = image_transform
        self.data = self.build_index()

    def build_index(self):
        samples = []
        for img_dir in sorted(self.image_root.iterdir()):
            img_folder = img_dir / "ring_front_center"
            lbl_folder = self.label_root / img_dir.name / "ring_front_center"

            images = sorted(img_folder.glob("*.jpg"))
            labels = {f.stem: f for f in lbl_folder.glob("*.txt")} if lbl_folder.exists() else {}

            for i in range(len(images)):
                img_prev = images[max(i - 1, 0)]
                img_curr = images[i]
                lbl_curr = labels.get(img_curr.stem, None)
                img_next = images[min(i + 1, len(images) - 1)]
                lbl_next = labels.get(img_next.stem, None)

                samples.append((img_prev, img_curr, img_next, lbl_next, lbl_curr))
        return samples

    def __len__(self):
        return len(self.data)

    def load_label(self, path, w, h):
        boxes, classes = [], []
        with open(path) as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.split())
                boxes.append([x, y, bw, bh])
                classes.append(int(c))
        return boxes, classes

    def __getitem__(self, idx):
        img_prev_p, img_curr_p, img_next_p, lbl_next_p, lbl_curr_p = self.data[idx]

        img_prev = cv2.cvtColor(cv2.imread(str(img_prev_p)), cv2.COLOR_BGR2RGB)
        img_curr = cv2.cvtColor(cv2.imread(str(img_curr_p)), cv2.COLOR_BGR2RGB)
        img_next = cv2.cvtColor(cv2.imread(str(img_next_p)), cv2.COLOR_BGR2RGB)

        h, w, _ = img_next.shape
        if lbl_curr_p is None:
            boxes_curr, classes_curr = [], []
        else:
            boxes_curr, classes_curr = self.load_label(lbl_curr_p, w, h)

        transformed_prev = self.image_transform(image=img_prev)
        img_prev = transformed_prev["image"]

        out_curr = self.box_transform(image=img_curr, bboxes=boxes_curr, class_labels=classes_curr)
        img_curr = out_curr["image"]
        boxes_curr = out_curr["bboxes"]
        classes_curr = out_curr["class_labels"]
        # (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]
        if len(boxes_curr):
            cls_curr = torch.tensor(
                [[c] for c in classes_curr],
                dtype=torch.float32,
            )
            bboxes_curr = torch.tensor(
                [[*b] for b in boxes_curr],
                dtype=torch.float32,
            )
            batch_idx_curr = torch.zeros(len(bboxes_curr), dtype=torch.long)
        else:
            cls_curr = torch.zeros((0,), dtype=torch.float32)
            bboxes_curr = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx_curr = torch.zeros((0,), dtype=torch.long)
            
        if lbl_next_p is None:
            boxes, classes = [], []
        else:
            boxes, classes = self.load_label(lbl_next_p, w, h)

        out = self.box_transform(image=img_next, bboxes=boxes, class_labels=classes)
        boxes = out["bboxes"]
        classes = out["class_labels"]
        # (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]
        if len(boxes):
            cls = torch.tensor(
                [[c] for c in classes],
                dtype=torch.float32,
            )
            bboxes = torch.tensor(
                [[*b] for b in boxes],
                dtype=torch.float32,
            )
            batch_idx = torch.zeros(len(bboxes), dtype=torch.long)
        else:
            cls = torch.zeros((0,), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx = torch.zeros((0,), dtype=torch.long)

        return {
            "img": torch.cat((img_curr, img_prev), dim=0),
            "cls_next": cls,
            "bboxes_next": bboxes,
            "batch_idx_next": batch_idx,
            "cls_curr": cls_curr,
            "bboxes_curr": bboxes_curr,
            "batch_idx_curr": batch_idx_curr,
        }


def collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch], dim=0)

    cls_curr, bboxes_curr, batch_idx_curr = [], [], []
    cls_next, bboxes_next, batch_idx_next = [], [], []

    for i, sample in enumerate(batch):

        # -------- Current frame --------
        if sample["bboxes_curr"].numel() > 0:
            cls_curr.append(sample["cls_curr"])
            bboxes_curr.append(sample["bboxes_curr"])
            batch_idx_curr.append(sample["batch_idx_curr"] + i)

        # -------- Next frame --------
        if sample["bboxes_next"].numel() > 0:
            cls_next.append(sample["cls_next"])
            bboxes_next.append(sample["bboxes_next"])
            batch_idx_next.append(sample["batch_idx_next"] + i)

    # ---------- Current ----------
    if len(cls_curr):
        cls_curr = torch.cat(cls_curr, dim=0)
        bboxes_curr = torch.cat(bboxes_curr, dim=0)
        batch_idx_curr = torch.cat(batch_idx_curr, dim=0)
    else:
        cls_curr = torch.zeros((0, 1), dtype=torch.float32)
        bboxes_curr = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx_curr = torch.zeros((0,), dtype=torch.long)

    # ---------- Next ----------
    if len(cls_next):
        cls_next = torch.cat(cls_next, dim=0)
        bboxes_next = torch.cat(bboxes_next, dim=0)
        batch_idx_next = torch.cat(batch_idx_next, dim=0)
    else:
        cls_next = torch.zeros((0, 1), dtype=torch.float32)
        bboxes_next = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx_next = torch.zeros((0,), dtype=torch.long)

    return {
        "img": imgs,

        "cls_curr": cls_curr,
        "bboxes_curr": bboxes_curr,
        "batch_idx_curr": batch_idx_curr,

        "cls_next": cls_next,
        "bboxes_next": bboxes_next,
        "batch_idx_next": batch_idx_next,
    }



