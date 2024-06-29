import numpy as np
import supervision  as sv
import cv2


import os
def get_min_max_bbox_xyxy(annotations: np.array):
    """.

    Parameters:
    annotations (np.array): annotation in xyxy

    Returns:
    np.array: New bounding box   representating the [xmin values, ymin values, xmax, ymax] of  all the boudning boxes
    """
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation

        min_x = min(min_x, xmin)
        min_y = min(min_y, ymin)
        max_x = max(max_x, xmax)
        max_y = max(max_y, ymax)


    return np.array([min_x, min_y, max_x, max_y])


def agg_annotations(annotations: sv.Detections, target_id=0):
    """
    Where we iterate through each annotation and aggreaget and create the min max bbox
    :param annotations: sv.Detections
    :param target_id:
    :return:
    """
    target_anns = []
    out_anns = sv.Detections.empty()
    ids = []
    bboxes = []
    for bbox, mask, conf, id, tracker_id, data in annotations:
        if id == target_id:
            target_anns.append(bbox)
        else:
            ids.append(int(id))
            bboxes.append(bbox)
    min_max_bbox = get_min_max_bbox_xyxy(target_anns)
    ids.append(target_id)
    bboxes.append(min_max_bbox)

    out_anns.xyxy = np.array(bboxes)
    out_anns.class_id = ids
    #need to explicity define conf  or it throws an error
    out_anns.confidence=None
    return out_anns

image_dir = "/Volumes/ColesSSD/Data/Vineyard/Canopy/Pinot-Noir/GrapesTrunks/images"
ann_dir = "/Volumes/ColesSSD/Data/Vineyard/Canopy/Pinot-Noir/GrapesTrunks/anns"
yaml_path = "/Volumes/ColesSSD/Data/Vineyard/Canopy/Pinot-Noir/GrapesTrunks/data.yaml"

ds = sv.DetectionDataset.from_yolo(images_directory_path=image_dir,
                                   annotations_directory_path=ann_dir,
                                   data_yaml_path=yaml_path)
new_anns = {}
image_dict = {}
out_image_dir = "/Users/cole/PycharmProjects/ann_converter/data/images"
out_ann_dir = "/Users/cole/PycharmProjects/ann_converter/data/anns"
out_yaml_dir = "/Users/cole/PycharmProjects/ann_converter/data/data.yaml"

for img_path in ds.images:
    # create new paths for the corresponding new ds
    out_img_path = os.path.join(out_image_dir, os.path.basename(img_path))
    out_ann_path = os.path.join(out_ann_dir, os.path.basename(img_path))

    anns = ds.annotations[img_path]
    min_max_box = agg_annotations(anns)

    # our anns and images in dict formate expected by the dataset object
    new_anns[out_img_path] = min_max_box
    #new image path where the key is the image path from the old ds and the value is the img_arr
    image_dict[out_img_path]= ds.images[img_path]

out_ds = sv.DetectionDataset(classes=ds.classes,
                         images=image_dict,
                         annotations=new_anns)

"""
plot random 
"""

#sv.plot_image(image)
out_ds.as_yolo(images_directory_path=out_image_dir,
               annotations_directory_path=out_ann_dir,
               data_yaml_path=out_yaml_dir)

new_ds = sv.DetectionDataset.from_yolo(images_directory_path=out_image_dir,
                                       annotations_directory_path=out_ann_dir,
                                       data_yaml_path=out_yaml_dir)
"""
plot random 
"""
import random
new_img_path = random.choice(list(new_ds.annotations.keys()))
img_arr = cv2.imread(new_img_path)
image = sv.BoundingBoxAnnotator(thickness=4).annotate(img_arr, new_ds.annotations[new_img_path])
# needed to change the code inSV was getting incorrect labels 
image = sv.LabelAnnotator(text_scale=2, text_thickness=4).annotate(image,
                                                                   new_ds.annotations[new_img_path])
                                                                   #labels=new_ds.classes)
sv.plot_image(image)
