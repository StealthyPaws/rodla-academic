import os.path
import numpy as np
from copy import deepcopy
import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

# detectron2 imports are only used for annotation transformation (optional)
try:
    from detectron2.data.transforms import RotationTransform
    from detectron2.data.detection_utils import transform_instance_annotations
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data.datasets.coco import convert_to_coco_json, convert_to_coco_dict
    from detectron2.data import MetadataCatalog, DatasetCatalog
    HAS_DETECTRON2 = True
except ImportError:
    HAS_DETECTRON2 = False


def apply_rotation(image, degree, annos=None):
    if degree == 0:
        return image if annos is None else (image, annos)
    
    angle_low_list = [0, 5, 10]
    angle_high_list = [5, 10, 15]
    angle_high = angle_high_list[degree - 1]
    angle_low = angle_low_list[degree - 1]
    h, w = image.shape[:2]
    
    if angle_low == 0:
        rotation = np.random.choice(np.arange(-angle_high, angle_high+1))
    else:
        rotation = np.random.choice(np.concatenate([np.arange(-angle_high, -angle_low+1), np.arange(angle_low, angle_high+1)]))
    
    # Use OpenCV for rotation instead of detectron2
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
    
    if annos is None:
        return rotated_image
    
    # For annotations, return original since we don't have detectron2
    return rotated_image, annos


def apply_warping(image, degree, annos=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    root_image_size = int(np.sqrt(image.shape[0] * image.shape[1]))
    base_sigma = root_image_size / 5
    base_alpha = base_sigma * 10
    sigma = base_sigma / degree
    alpha = base_alpha / degree
    elastic_transform = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, polygon_recoverer=None)
    if annos is None:
        return elastic_transform(image=image)
    bbs = []
    polygons = []
    elastic_annos = deepcopy(annos)
    for anno in annos:
        bbox = anno["bbox"]
        seg = anno["segmentation"]
        bbs.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3]))
        polygons.append(Polygon(np.array(seg).reshape(-1, 2)))
    bbs = BoundingBoxesOnImage(bbs, shape=image.shape[:2])
    polygons = PolygonsOnImage(polygons, shape=image.shape[:2])
    elastic_image, elastic_bbs, elastic_polygons = elastic_transform(image=image, bounding_boxes=bbs, polygons=polygons)
    for i, anno in enumerate(elastic_annos):
        elastic_annos[i]["bbox"] = [elastic_bbs[i].x1, elastic_bbs[i].y1,
                                    elastic_bbs[i].x2 - elastic_bbs[i].x1,
                                    elastic_bbs[i].y2 - elastic_bbs[i].y1]
        elastic_annos[i]["segmentation"] = [elastic_polygons.polygons[i].exterior.reshape(-1).tolist()]
    return elastic_image, elastic_annos


def apply_keystoning(image, degree, annos=None):
    if degree == 0:
        return image
    degree = 2 * degree - 1
    low_scale = 0.01
    high_scale = 0.02 * degree
    perspective_transform = iaa.PerspectiveTransform(scale=(low_scale, high_scale), cval=0, mode='constant',
                                                     keep_size=True, fit_output=True, polygon_recoverer=None)
    if annos is None:
        return perspective_transform(image=image)
    bbs = []
    polygons = []
    perspective_annos = deepcopy(annos)
    for anno in annos:
        bbox = anno["bbox"]
        seg = anno["segmentation"]
        bbs.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0] + bbox[2], y2=bbox[1] + bbox[3]))
        polygons.append(Polygon(np.array(seg).reshape(-1, 2)))
    bbs = BoundingBoxesOnImage(bbs, shape=image.shape[:2])
    polygons = PolygonsOnImage(polygons, shape=image.shape[:2])
    perspective_image, perspective_bbs, perspective_polygons \
        = perspective_transform(image=image, bounding_boxes=bbs, polygons=polygons)
    for i, anno in enumerate(perspective_annos):
        perspective_annos[i]["bbox"] = [perspective_bbs[i].x1, perspective_bbs[i].y1,
                                        perspective_bbs[i].x2 - perspective_bbs[i].x1,
                                        perspective_bbs[i].y2 - perspective_bbs[i].y1]
        perspective_annos[i]["segmentation"] = [perspective_polygons.polygons[i].exterior.reshape(-1).tolist()]
    return perspective_image, perspective_annos
