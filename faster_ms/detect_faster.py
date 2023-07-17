import os
import cv2 
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from src.config import config
# from detectron2.data.datasets import register_coco_instances
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_small2017_dicts():
    from pycocotools.coco import COCO

    coco_root = config.coco_root
    data_type = config.train_data_type

    #Classes need to train or test.
    train_cls = config.coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}

    
    dataset_dicts = []
    skipped = 0
    idx = 0
    for img_id in image_ids:
        record = {}
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        if not os.path.exists(image_path):
            # print("skipped, ", image_path)
            skipped += 1
            # if skipped > 10:
            #     return dataset_dicts
            continue

        height, width = cv2.imread(image_path).shape[:2]
        
        record["file_name"] = image_path 
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        objs = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            if not class_name in train_cls:
                continue
            x1, x2 = bbox[0], bbox[0] + bbox[2]
            y1, y2 = bbox[1], bbox[1] + bbox[3]

            obj = {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": train_cls_dict[class_name],
                "iscrowd": int(label["iscrowd"])
            }
            objs.append(obj)

        record["annotations"] = objs
        # print(record)
        dataset_dicts.append(record)
        idx += 1
        # if idx > 3:
        #     break
    return dataset_dicts


for d in ["train"]:
    DatasetCatalog.register("small2017_" + d, lambda d=d: get_small2017_dicts())
    MetadataCatalog.get("small2017_" + d).set(thing_classes=["small2017"])
small2017_metadata = MetadataCatalog.get("small2017_train")

# register_coco_instances("train2017small", {}, "/home/jyan/tmp/coco/annotations/instances_train2017.json",
#         "/home/jyan/tmp/coco/train2017")

cfg = get_cfg()
cfg.merge_from_file("/home/jyan/tmp/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# faster_rcnn_R_50_FPN_3x.yaml
cfg.DATASETS.TRAIN = ("small2017_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = (
   60 
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
   4096 
)  # faster, and good enough for this toy dataset , default 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
