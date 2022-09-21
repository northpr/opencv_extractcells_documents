import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List
from detect import detect_box
import numpy as np
import cv2
from tqdm.auto import tqdm



@dataclass
class BoxAnnotation:
    # this is a common data structure we use in many projects, pls use it as it is.
    x: int  # upper left corner x (absolute)
    y: int  # upper left corner y (absolute)
    width: int  # box width (absolute)
    height: int  # box height (absolute)
    class_name: str  # identifier for row and column starting at 0 (format: f"cell-{row}-{col}")


class TableAnalysis:

    config = {
        "parameter1": 42,
        "HOUGHLINES":{
            "RHO": 1,
            "THETA": np.pi/180,
            "THRESHOLD": 15, # For delete
            "HORIZONTAL_MIN_LINE_LENGTH": 50,  # minimum  pixels making up a line (recommend at 50)
            "VERTICAL_MIN_LINE_LENGTH": 600,  # minimum pixels making up a line (recommend at 700)
            "MAX_LINE_GAP": 50},
        "PERCENT_BOX_MAX_WIDTH": 0.4,  # Size of the box to percent depends on the size of image 0 ~ 1 (recommend at 0.4)
        "PERCENT_BOX_MIN_WIDTH": 0.1,  # 0 ~ 1 (recommend at 0.1)
        "PERCENT_BOX_MAX_HEIGHT": 0.5,  # 0 ~ 1 (recommend at 0.5)
        "PERCENT_BOX_MIN_HEIGHT": 0.03  # 0 ~ 1 (recommend at 0.03)
        # you should insert all config parameters here -> we will be able to read and change it with our framework
    }

    def process(self, filepath: Path) -> List[BoxAnnotation]:

        box_annotations: List[BoxAnnotation] = []

        # TODO: Insert your code (only) here, you can also write it in separate files and call it from here
        image = cv2.imread(str(filepath))
        stats, labels = detect_box(image, self.config)
        image_width = image.shape[0]
        image_height = image.shape[1]
        
        # example which creates one cell that covers the whole image
        for row, stat in enumerate(stats):
            col = 0
            for x_axis, y_axis, w, h, area in stat:
                if image_width*self.config["PERCENT_BOX_MAX_WIDTH"] > w > image_width*self.config["PERCENT_BOX_MIN_WIDTH"] \
                    and image_height*self.config["PERCENT_BOX_MAX_HEIGHT"] > h > image_height*self.config["PERCENT_BOX_MIN_HEIGHT"]:
                    box_annotations.append(BoxAnnotation(x=x_axis, y=y_axis , width=w, height=h, class_name=f"cell-{row}-{col}"))
                    col += 1
                    
        return box_annotations

    def write_results(self, box_annotations: List[BoxAnnotation], filepath: Path, output_dir: Path):
        # this function should also not be changed

        # create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        image_height: int
        image_width: int
        image_height, image_width, *_ = image.shape

        for annotation in box_annotations:
            x1 = max(0, annotation.x)
            y1 = max(0, annotation.y)
            x2 = min(image_width, x1 + annotation.width)
            y2 = min(image_height, y1 + annotation.height)

            cell = image[y1:y2, x1:x2, ...]

            cv2.imwrite(f"{output_dir/annotation.class_name}.png", cell)


def main(table_dir: str, result_dir: str):
    # this part calls the process function above for every picture (pls. dont change it)
    table_analysis = TableAnalysis()

    tables = list(Path(table_dir).glob("*.png"))

    filepath: Path
    for filepath in tqdm(tables, desc="Processing Tables", unit="tables"):
        box_annotations = table_analysis.process(filepath)
        output_dir = Path(result_dir, filepath.stem)
        table_analysis.write_results(box_annotations, filepath, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_dir", type=str, nargs="?", default="./tables")
    parser.add_argument("result_dir", type=str, nargs="?", default="./results")
    args = parser.parse_args()
    main(table_dir=args.table_dir, result_dir=args.result_dir)
