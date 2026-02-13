import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)

def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    #raise NotImplementedError("Not implemented")

    with open(info_path) as f:
        info = json.load(f)

    detections_all = info.get("detections", [])
    if view_index >= len(detections_all):
        return []

    frame_detections = detections_all[view_index]

    # Build track_id -> kart_name mapping
    id_to_name = {}

    # Case 1: explicit dict mapping
    if isinstance(info.get("kart_id_to_name"), dict):
        for k, v in info["kart_id_to_name"].items():
            try:
                id_to_name[int(k)] = str(v)
            except Exception:
                continue

    # Case 2: karts as LIST OF STRINGS
    elif isinstance(info.get("karts"), list) and info["karts"]:
        if isinstance(info["karts"][0], str):
            id_to_name = {i: name for i, name in enumerate(info["karts"])}

        # Case 3: karts as list of dicts
        elif isinstance(info["karts"][0], dict):
            for k in info["karts"]:
                if "id" in k and ("name" in k or "kart_name" in k):
                    try:
                        kid = int(k["id"])
                        id_to_name[kid] = str(k.get("name", k.get("kart_name")))
                    except Exception:
                        continue

    # Case 4: alternate dict format
    elif isinstance(info.get("kart_names"), dict):
        for k, v in info["kart_names"].items():
            try:
                id_to_name[int(k)] = str(v)
            except Exception:
                continue

    # Scale detection boxes to resized image space
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []

    for det in frame_detections:
        if len(det) != 6:
            continue

        class_id, track_id, x1, y1, x2, y2 = det
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:  # Only karts
            continue

        # Scale coordinates
        x1s = x1 * scale_x
        y1s = y1 * scale_y
        x2s = x2 * scale_x
        y2s = y2 * scale_y

        # Skip tiny boxes
        if (x2s - x1s) < min_box_size or (y2s - y1s) < min_box_size:
            continue

        # Skip if fully outside image
        if x2s < 0 or x1s > img_width or y2s < 0 or y1s > img_height:
            continue

        cx = (x1s + x2s) / 2
        cy = (y1s + y2s) / 2

        kart_name = id_to_name.get(track_id, str(track_id))

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_name,
                "center": (cx, cy),
                "is_center_kart": False,
            }
        )

    if not karts:
        return []

    # Identify ego kart
    ego_idx = None

    # Prefer track_id == 0
    for i, k in enumerate(karts):
        if k["instance_id"] == 0:
            ego_idx = i
            break

    # If not found, pick closest to image center
    if ego_idx is None:
        img_cx, img_cy = img_width / 2, img_height / 2
        ego_idx = min(
            range(len(karts)),
            key=lambda i: (
                (karts[i]["center"][0] - img_cx) ** 2
                + (karts[i]["center"][1] - img_cy) ** 2
            ),
        )

    karts[ego_idx]["is_center_kart"] = True

    return karts



def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    #raise NotImplementedError("Not implemented")

    """
    Extract track information from the info.json file.
    """
    with open(info_path) as f:
        info = json.load(f)

    # common key: "track"
    if "track" in info and isinstance(info["track"], str):
        return info["track"]

    # fallback keys
    for k in ["track_name", "trackName", "map", "level"]:
        if k in info and isinstance(info[k], str):
            return info[k]

    return "unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    #raise NotImplementedError("Not implemented")

    karts = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not karts:
        return []

    ego = next((k for k in karts if k.get("is_center_kart", False)), karts[0])
    track_name = extract_track_info(info_path)

    qa_pairs = []

    # 1) Ego car
    qa_pairs.append({"question": "What kart is the ego car?", "answer": str(ego["kart_name"])})

    # 2) Total karts
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})

    # 3) Track
    qa_pairs.append({"question": "What track is this?", "answer": str(track_name)})

    # 4) Relative position for each other kart (front/behind)
    # smaller y is higher in image (more "front")
    ego_y = float(ego["center"][1])
    for k in karts:
        if k is ego or k.get("is_center_kart", False):
            continue
        other_y = float(k["center"][1])
        pos = "front" if other_y < ego_y else "back"
        qa_pairs.append(
            {
                "question": f"Is {k['kart_name']} in front of or behind the ego car?",
                "answer": pos,
            }
        )

    # 5) How many in front
    num_front = sum(
        1
        for k in karts
        if not k.get("is_center_kart", False) and float(k["center"][1]) < ego_y
    )
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(num_front)})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def build_qa_dataset(
    split: str = "train",
    out_name: str = "generated",
    data_dir: str = "../data",
    max_info_files: int | None = None,
):
    """
    Build a *_qa_pairs.json file under data/<split>/ by looping over *_info.json
    and all available view indices, then attaching image_file paths.
    """
    data_dir = Path(__file__).parent / data_dir
    split_dir = data_dir / split

    info_files = sorted(split_dir.glob("*_info.json"))
    if max_info_files is not None:
        info_files = info_files[:max_info_files]

    qa_pairs_all: list[dict] = []

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")  # e.g. "00000"
        # dataset has views 00..09 typically
        for view_index in range(10):
            image_candidates = list(split_dir.glob(f"{base}_{view_index:02d}_im.jpg"))
            if not image_candidates:
                continue

            # store path relative to data/ (matches how VQADataset builds image_path)
            image_file_rel = f"{split}/{image_candidates[0].name}"

            qa_list = generate_qa_pairs(str(info_file), view_index)
            for qa in qa_list:
                qa_pairs_all.append(
                    {
                        "image_file": image_file_rel,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

    out_path = split_dir / f"{out_name}_qa_pairs.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs_all, f, indent=2)

    print(f"Wrote {len(qa_pairs_all)} QA pairs to {out_path}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "build": build_qa_dataset})


if __name__ == "__main__":
    main()
