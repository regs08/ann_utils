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


def min_max_annotations(annotations: sv.Detections, target_id=0):
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


