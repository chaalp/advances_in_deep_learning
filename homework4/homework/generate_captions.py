from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info

import json

'''
def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    #raise NotImplementedError("Not implemented")

    """
    Generate caption(s) for a specific view.
    Captions mirror the QA semantics with atomic statements.
    """
    # reuse QA helpers from generate_qa
    from .generate_qa import extract_kart_objects, extract_track_info

    karts = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not karts:
        return []

    ego = next((k for k in karts if k.get("is_center_kart", False)), karts[0])
    track_name = extract_track_info(info_path)

    captions = []

    # 1) Ego
    captions.append(f"{ego['kart_name']} is the ego car.")

    # 2) Counting (match demo phrasing)
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3) Track
    captions.append(f"The track is {track_name}.")

    # 4) Relative position for each non-ego kart
    ego_y = float(ego["center"][1])
    for k in karts:
        if k is ego or k.get("is_center_kart", False):
            continue
        pos = "in front of" if float(k["center"][1]) < ego_y else "behind"
        captions.append(f"{k['kart_name']} is {pos} of the ego car.")

    # 5) Count in front (extra useful signal)
    num_front = sum(
        1 for k in karts if not k.get("is_center_kart", False) and float(k["center"][1]) < ego_y
    )
    captions.append(f"There are {num_front} karts in front of the ego car.")

    return captions
'''

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list[str]:
    """
    Generate caption(s) for a specific view.

    IMPORTANT: These captions are designed to match the MultiChoiceQADataset
    candidate templates EXACTLY (e.g., "scene" not "scenario", "behind the ego car",
    "in front of the ego car", "left/right of the ego car").
    """
    from .generate_qa import extract_kart_objects, extract_track_info

    karts = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not karts:
        return []

    ego = next((k for k in karts if k.get("is_center_kart", False)), karts[0])
    track_name = extract_track_info(info_path)

    captions: list[str] = []

    # --- helpers ---
    ego_x = float(ego["center"][0])
    ego_y = float(ego["center"][1])

    # Small deadzone so tiny pixel jitter doesn't flip left/right/front/back
    EPS_X = 3.0
    EPS_Y = 3.0

    def is_front(k) -> bool:
        return float(k["center"][1]) < ego_y - EPS_Y

    def is_behind(k) -> bool:
        return float(k["center"][1]) > ego_y + EPS_Y

    def is_left(k) -> bool:
        return float(k["center"][0]) < ego_x - EPS_X

    def is_right(k) -> bool:
        return float(k["center"][0]) > ego_x + EPS_X

    others = [k for k in karts if not k.get("is_center_kart", False) and k is not ego]

    # 1) Ego (matches candidate template)
    captions.append(f"{ego['kart_name']} is the ego car.")

    # 2) Total count (MATCH: "scene")
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3) Track (matches candidate template)
    captions.append(f"The track is {track_name}.")

    # 4) Per-kart single relations (matches candidate templates)
    for k in others:
        name = k["kart_name"]

        # front/back
        if is_front(k):
            captions.append(f"{name} is in front of the ego car.")
        elif is_behind(k):
            captions.append(f"{name} is behind the ego car.")

        # left/right
        if is_left(k):
            captions.append(f"{name} is left of the ego car.")
        elif is_right(k):
            captions.append(f"{name} is right of the ego car.")

    # 5) Count captions (optional but helpful and candidate-aligned)
    captions.append(count_sentence(sum(is_front(k) for k in others), "in front of the ego car"))
    captions.append(count_sentence(sum(is_behind(k) for k in others), "behind the ego car"))
    captions.append(count_sentence(sum(is_left(k) for k in others), "to the left of the ego car"))
    captions.append(count_sentence(sum(is_right(k) for k in others), "to the right of the ego car"))

    return captions

def count_sentence(n: int, relation: str) -> str:
    if n == 1:
        return f"There is 1 kart {relation}."
    else:
        return f"There are {n} karts {relation}."

def build_captions_dataset(
    split: str = "train",
    out_name: str = "generated",
    data_dir: str = "../data",
    max_info_files: int | None = None,
):
    """
    Build a *_captions.json file under data/<split>/ by looping over *_info.json
    and all available view indices, then attaching image_file paths.
    """
    data_dir = Path(__file__).parent / data_dir
    split_dir = data_dir / split

    info_files = sorted(split_dir.glob("*_info.json"))
    if max_info_files is not None:
        info_files = info_files[:max_info_files]

    all_pairs: list[dict] = []

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_candidates = list(split_dir.glob(f"{base}_{view_index:02d}_im.jpg"))
            if not image_candidates:
                continue

            image_file_rel = f"{split}/{image_candidates[0].name}"

            captions = generate_caption(str(info_file), view_index)
            for cap in captions:
                all_pairs.append({"image_file": image_file_rel, "caption": cap})

    out_path = split_dir / f"{out_name}_captions.json"
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Wrote {len(all_pairs)} captions to {out_path}")

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "build": build_captions_dataset})


if __name__ == "__main__":
    main()
