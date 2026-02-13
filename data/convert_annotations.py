"""
Annotation format converters.

Converts between common annotation formats used by different labeling tools
and the formats expected by our training scripts.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR


def create_yolo_dataset_yaml(
    name: str,
    train_images: str,
    val_images: str,
    classes: list[str],
    test_images: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Create a YOLO dataset YAML config file.

    Args:
        name:         Dataset name.
        train_images: Path to training images directory.
        val_images:   Path to validation images directory.
        classes:      List of class names.
        test_images:  Optional path to test images directory.
        output_path:  Where to save the YAML (default: data/<name>.yaml).

    Returns:
        Path to the created YAML file.
    """
    dataset_config = {
        "path": str(DATA_DIR / name),
        "train": train_images,
        "val": val_images,
        "names": {i: cls_name for i, cls_name in enumerate(classes)},
    }
    if test_images:
        dataset_config["test"] = test_images

    output_path = output_path or str(DATA_DIR / f"{name}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"Created YOLO dataset config: {output_path}")
    return output_path


def coco_to_yolo(
    coco_json_path: str,
    output_labels_dir: str,
    target_classes: list[str] | None = None,
) -> None:
    """
    Convert COCO-format annotations to YOLO-format label files.

    COCO format: { "images": [...], "annotations": [...], "categories": [...] }
    YOLO format: one .txt per image, each line = class_id cx cy w h (normalized)

    Args:
        coco_json_path:    Path to the COCO JSON annotation file.
        output_labels_dir: Directory to save YOLO .txt label files.
        target_classes:    Only convert these class names (None = all).
    """
    output_dir = Path(output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build lookup maps
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_map = {
        img["id"]: {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }
        for img in coco["images"]
    }

    # Filter categories if needed
    if target_classes:
        valid_cat_ids = {
            cid for cid, name in cat_map.items() if name in target_classes
        }
        class_to_idx = {name: i for i, name in enumerate(target_classes)}
    else:
        valid_cat_ids = set(cat_map.keys())
        all_names = sorted(set(cat_map.values()))
        class_to_idx = {name: i for i, name in enumerate(all_names)}

    # Group annotations by image
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in valid_cat_ids:
            img_annotations.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    for img_id, img_info in img_map.items():
        anns = img_annotations.get(img_id, [])
        label_name = Path(img_info["file_name"]).stem + ".txt"
        label_path = output_dir / label_name

        w_img, h_img = img_info["width"], img_info["height"]
        lines = []
        for ann in anns:
            cat_name = cat_map[ann["category_id"]]
            cls_id = class_to_idx[cat_name]
            x, y, w, h = ann["bbox"]  # COCO: top-left x,y + width,height

            # Convert to YOLO: center_x, center_y, width, height (normalized)
            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            nw = w / w_img
            nh = h / h_img
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    print(f"Converted {converted} images from COCO → YOLO format in {output_dir}")


def voc_to_yolo(
    voc_dir: str,
    output_labels_dir: str,
    classes: list[str],
) -> None:
    """
    Convert Pascal VOC XML annotations to YOLO format.

    Args:
        voc_dir:           Directory containing VOC .xml files.
        output_labels_dir: Directory to save YOLO .txt label files.
        classes:           Ordered list of class names.
    """
    import xml.etree.ElementTree as ET

    voc_dir = Path(voc_dir)
    output_dir = Path(output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: i for i, name in enumerate(classes)}
    converted = 0

    for xml_file in sorted(voc_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        w_img = int(size.find("width").text)
        h_img = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_to_idx:
                continue

            cls_id = class_to_idx[name]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            cx = ((x1 + x2) / 2) / w_img
            cy = ((y1 + y2) / 2) / h_img
            nw = (x2 - x1) / w_img
            nh = (y2 - y1) / h_img
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = output_dir / (xml_file.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    print(f"Converted {converted} VOC annotations → YOLO format in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation format converters")
    sub = parser.add_subparsers(dest="command")

    # COCO → YOLO
    coco_parser = sub.add_parser("coco2yolo", help="Convert COCO JSON to YOLO txt")
    coco_parser.add_argument("coco_json", help="Path to COCO annotations JSON")
    coco_parser.add_argument("output_dir", help="Output directory for YOLO .txt files")
    coco_parser.add_argument("--classes", nargs="+", default=None, help="Filter to these classes")

    # VOC → YOLO
    voc_parser = sub.add_parser("voc2yolo", help="Convert VOC XML to YOLO txt")
    voc_parser.add_argument("voc_dir", help="Directory with VOC .xml files")
    voc_parser.add_argument("output_dir", help="Output directory for YOLO .txt files")
    voc_parser.add_argument("--classes", nargs="+", required=True, help="Ordered class names")

    # Create YAML
    yaml_parser = sub.add_parser("create_yaml", help="Create YOLO dataset YAML")
    yaml_parser.add_argument("name", help="Dataset name")
    yaml_parser.add_argument("--train", required=True, help="Train images path")
    yaml_parser.add_argument("--val", required=True, help="Val images path")
    yaml_parser.add_argument("--test", default=None, help="Test images path")
    yaml_parser.add_argument("--classes", nargs="+", required=True, help="Class names")

    args = parser.parse_args()

    if args.command == "coco2yolo":
        coco_to_yolo(args.coco_json, args.output_dir, args.classes)
    elif args.command == "voc2yolo":
        voc_to_yolo(args.voc_dir, args.output_dir, args.classes)
    elif args.command == "create_yaml":
        create_yolo_dataset_yaml(args.name, args.train, args.val, args.classes, args.test)
    else:
        parser.print_help()
