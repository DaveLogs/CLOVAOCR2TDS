# CLOVAOCR2TDS
Convert Naver CLOVA AI OCR format data to training-datasets-splitter format data.

## References

- [CLOVA OCR API Documents](https://apidocs.ncloud.com/ko/ai-application-service/ocr/ocr/): Naver CLOVA AI OCR
- [CLOVA OCR User Guide](https://davelogs.tistory.com/38): Naver CLOVA General OCR user guide
- [training-datasets-splitter](https://github.com/DaveLogs/training-datasets-splitter): Split the training datasets into 'training'/'validation'/'test' data.

## Usage example

```bash
(venv) $ python3 convert.py \
                --input_path ./input \
                --output_path "./output" \
                --clova_api_url "YOUR_API_URL" \
                --clova_secret_key "YOUR_SECRET_KEY"
```


## Input Data Structures

The structure of input data folder as below.

* Input: image to be recognized 

```
/input
#   [id].[ext]
├── 0000000001.png
├── 0000000002.png
├── 0000000003.png
└── ...
```


## Output Data Structure

The structure of output data folder as below.

* Output: for use in [training-datasets-splitter](https://github.com/DaveLogs/training-datasets-splitter) project.

```
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
```

### Label file structure

* labels.txt

```
# {filename}\t{label}\n
  0000000001_001.png	abcd
  0000000001_002.png	efgh
  ...
  0000000002_001.png	ijkl
  0000000002_002.png	mnop
  0000000002_003.png	qrst
  ...
```

### JSON file structure for LabelMe

refer to the [LabelMe GitHub Repository](https://github.com/wkentaro/labelme).