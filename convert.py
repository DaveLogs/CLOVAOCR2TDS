"""

Convert Naver CLOVA AI OCR format data to training-datasets-splitter format data.

## References

1. CLOVA OCR API Documents: https://apidocs.ncloud.com/ko/ai-application-service/ocr/ocr/
2. CLOVA OCR User Guide: https://davelogs.tistory.com/38
3. training-datasets-splitter: https://github.com/DaveLogs/training-datasets-splitter

## Usage example:
    python3 convert.py \
            --input_path "./input" \
            --output_path "./output" \
            --clova_api_url "YOUR_API_URL" \
            --clova_secret_key "YOUR_SECRET_KEY"

## Input data structure:

    /input
    #   [id].[ext]
    ├── 0000000001.png
    ├── 0000000002.png
    ├── 0000000003.png
    └── ...

## Output data structure:

    /output
    ├── /recognized
    │   #   The recognized result using CLOVA General OCR
    │   ├── 0000000001_clova.json
    │   ├── 0000000002_clova.json
    │   └── ...
    │
    ├── /cropped
    │   #   cropped image and label
    │   ├── 0000000001_001.png
    │   ├── 0000000001_002.png
    │   ├── ...
    │   ├── 0000000002_001.png
    │   ├── 0000000002_002.png
    │   ├── 0000000002_003.png
    │   ├── ...
    │   └── labels.txt
    │
    └── /converted
        #   image & json for LabelMe
        ├── 0000000001.png
        ├── 0000000001.json
        ├── 0000000002.png
        ├── 0000000002.json
        └── ...

* Label file structure:

    # {filename}\t{label}\n
      0000000001_001.png	abcd
      0000000001_002.png	efgh
      ...
      0000000002_001.png	ijkl
      0000000002_002.png	mnop
      0000000002_003.png	qrst
      ...

"""

import os
import sys
import time
import json
import uuid
import shutil
import argparse
import requests

from PIL import Image
from collections import OrderedDict


def run(args):
    """ Convert Naver CLOVA AI OCR format data to training-datasets-splitter format data """

    if not os.path.exists(args.input_path):
        sys.exit(f"Can't find '{os.path.abspath(args.input_path)}' directory.")

    if os.path.isdir(args.output_path):
        sys.exit(f"'{os.path.abspath(args.output_path)}' directory is already exists.")
        # print(f"'{os.path.abspath(args.output_path)}' directory is already exists.")
    else:
        # dirs[0]: root, dirs[1]: CLOVA OCR result, dirs[2]: cropped, dirs[3]: converted (for LabelMe)
        dirs = create_working_directory(args.output_path, ["recognized", "cropped", "converted"])

    files, count = get_files(args.input_path)

    labels = open(os.path.join(args.output_path, dirs[2], "labels.txt"), "w", encoding="utf8")

    start_time = time.time()
    digits = len(str(count))
    for ii, file_name in enumerate(files):
        if (ii + 1) % 10 == 0:
            print(("\r%{}d / %{}d Processing !!".format(digits, digits)) % (ii + 1, count), end="")

        name, ext = file_name.split('.')

        clova_json_file = request_recognition_from_clova_ocr(args, dirs[1], file_name)
        # clova_json_file = f"{dirs[1]}/{name}_clova.json"
        print(f"clova_json: {clova_json_file}")

        with open(os.path.join(clova_json_file)) as f:
            json_data = json.load(f)

        json_dict = OrderedDict()

        json_dict["version"] = "4.5.9"
        json_dict["shape_type"] = "rectangle"
        json_dict["flags"] = {}
        shapes = []

        with Image.open(os.path.join(args.input_path, file_name)) as img:
            for jj, fields in enumerate(json_data["images"][0]["fields"]):
                label = fields["inferText"]
                bbox = get_bbox(fields["boundingPoly"]["vertices"])
                # print(f"label: {label}, bbox: {bbox}")
                if not valid_crop_size(bbox, args.min_image_size):
                    # print(f"'{file_name}' - invalid bbox: {bbox}")
                    continue

                # save cropped image and label
                cropped_image = img.crop(bbox)
                cropped_file = f"{name}_{jj:03d}.{ext}"
                cropped_image.save(os.path.join(args.output_path, dirs[2], cropped_file))
                labels.write(f"{cropped_file}\t{label}\n")

                shapes_dict = OrderedDict()
                shapes_dict["label"] = label
                shapes_dict["points"] = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
                shapes_dict["group_id"] = None
                shapes_dict["shape_type"] = "rectangle"
                shapes_dict["flags"] = {}
                shapes.append(shapes_dict)

            json_dict["shapes"] = shapes
            json_dict["imagePath"] = file_name
            json_dict["imageData"] = None
            json_dict["imageHeight"] = img.size[1]
            json_dict["imageWidth"] = img.size[0]

        # save json (labelme format)
        shutil.copy(os.path.join(args.input_path, file_name), os.path.join(dirs[3], file_name))
        with open(os.path.join(dirs[3], name + ".json"), 'w', encoding='utf-8') as outfile:
            json.dump(json_dict, outfile, ensure_ascii=False, indent="\t")

    labels.close()

    elapsed_time = (time.time() - start_time) / 60.
    print("\n- processing time: %.1fmin" % elapsed_time)


def request_recognition_from_clova_ocr(args, subdir, image_file):
    name, ext = image_file.split('.')
    request_json = {
        'images': [
            {
                'format': ext,
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', open(os.path.join(args.input_path, image_file), 'rb'))
    ]
    headers = {
        'X-OCR-SECRET': args.clova_secret_key
    }

    response = requests.request("POST", args.clova_api_url, headers=headers, data=payload, files=files)

    res = json.loads(response.text.encode('utf8'))
    # print(res)

    json_file = os.path.join(args.output_path, subdir, name + "_clova.json")
    # print(f"json_file: {json_file}")
    with open(json_file, 'w', encoding='utf-8') as outfile:
        json.dump(res, outfile, indent=4, ensure_ascii=False)

    return json_file


def get_bbox(points):
    left = min(p["x"] for p in points)  # left = points[0]["x"]
    upper = min(p["y"] for p in points)  # upper = points[0]["y"]
    right = max(p["x"] for p in points)  # right = points[2]["x"]
    lower = max(p["y"] for p in points)  # lower = points[2]["y"]

    bbox = [left, upper, right, lower]
    # print(f"bbox: {bbox}")

    return bbox


def valid_crop_size(bbox, size):
    if bbox[2] - bbox[0] < size or bbox[3] - bbox[1] < size:
        return False

    return True


def get_files(path, except_file=""):
    file_list = []

    for file in os.listdir(path):
        if file.startswith(".") or file == os.path.basename(except_file):
            print('except file name: ', file)
            continue

        file_list.append(file)

    file_list.sort()

    return file_list, len(file_list)


def create_working_directory(root, sub_dirs=None):
    dirs = [root]
    os.makedirs(root)
    for sub in sub_dirs:
        path = os.path.join(root, sub)
        dirs.append(path)
        os.makedirs(path)

    return dirs


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert dataset for training-datasets-splitter')

    parser.add_argument('--input_path', type=str, required=True, help='Data path of images to be recognized')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Data path for use in training-datasets-splitter project')
    # parser.add_argument('--clova_api_url', type=str, required=True, help='Your CLOVA OCR API URL')
    # parser.add_argument('--clova_secret_key', type=str, required=True, help='Your CLOVA OCR secret key')
    parser.add_argument('--min_image_size', type=int, default=16, help='The minimum size of the cropped image')

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    arguments = parse_arguments()

    arguments.clova_api_url = "YOUR_API_URL"
    arguments.clova_secret_key = "YOUR_SECRET_KEY"

    run(arguments)

