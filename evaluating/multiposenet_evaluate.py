import os
import sys
import cv2
import numpy as np
import json
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.keypoint_joint_utils import get_joint_list
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import math
from utils.prn_gaussian import gaussian, crop
from network.prn_network import *
from network.posecnet import PoseCNet
import argparse



def coco_eval(coco_dir, backbone="resnet50", filename_result="ann_coco_result.json", write_json=False):

    coco_val = os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json')
    coco = COCO(coco_val)
    img_ids = coco.getImgIds(catIds=[1])

    posecnet = PoseCNet(bck_arch=backbone)
    posecnet.load_subnet_weights(k_weights="../Models/model.85-86.60.hdf5",
                                 d_weights="../Models/inference_detection_resnet50_0.421.h5")
                                 #p_weights="../Models/prn_epoch20_final.h5"

    multipose_results = []
    coco_order = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]

    for img_id in tqdm(img_ids):

        img_name = coco.loadImgs(img_id)[0]['file_name']

        oriImg = cv2.imread(os.path.join(coco_dir, 'val2017/', img_name)).astype(np.float32)
        multiplier = get_multiplier(oriImg)

        # Get results of original image
        orig_heat, orig_bbox_all = get_outputs(multiplier, oriImg, posecnet.model)

        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_heat, flipped_bbox_all = get_outputs(multiplier, swapped_img, posecnet.model)

        # compute averaged heatmap
        heatmaps = handle_heat(orig_heat, flipped_heat)

        # segment_map = heatmaps[:, :, 17]
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        joint_list = get_joint_list(oriImg, param, heatmaps[:, :, :18], 1)
        joint_list = joint_list.tolist()

        joints = []
        for joint in joint_list:
            if int(joint[-1]) != 1:
                joint[-1] = max(0, int(joint[-1]) - 1)
            joints.append(joint)
        joint_list = joints

        prn_result = prn_network(joint_list, orig_bbox_all[1], img_name, img_id)
        for result in prn_result:
            keypoints = result['keypoints']
            del keypoints[3:6] #delete neck points
            coco_keypoint = []
            for i in range(17):
                coco_keypoint.append(keypoints[coco_order[i] * 3])
                coco_keypoint.append(keypoints[coco_order[i] * 3 + 1])
                coco_keypoint.append(keypoints[coco_order[i] * 3 + 2])
            result['keypoints'] = coco_keypoint
            multipose_results.append(result)

    ann_filename = filename_result
    with open(ann_filename, "w") as f:
        json.dump(multipose_results, f, indent=4)
    # load results in COCO evaluation tool
    coco_pred = coco.loadRes(ann_filename)
    # run COCO evaluation
    coco_eval = COCOeval(coco, coco_pred, 'keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if not write_json:
        os.remove(ann_filename)


def get_multiplier(img):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    scale_search = [0.5, 1., 1.5, 2, 2.5]
    return [x * 480 / float(img.shape[0]) for x in scale_search]

def get_outputs(multiplier, img, model):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 18))
    bbox_all = []
    # max_scale = multiplier[-1]
    # max_size = max_scale * img.shape[0]
    # # padding
    # max_cropped, _, _ = crop_with_factor(
    #     img, max_size, factor=32)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = crop_with_factor(
            img, inp_size, factor=32, pad_val=128)


        im_data = np.expand_dims(im_cropped, 0)


        heatmaps, boxes, scores, labels = model.predict(im_data)
        boxes = boxes[0]
        scores = scores[0]

        heatmap = heatmaps[0, :int(im_cropped.shape[0] / 4), :int(im_cropped.shape[1] / 4), :18]
        heatmap = cv2.resize(heatmap, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(
            heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # bboxs
        idxs = np.where(scores > 0.4)
        bboxs = []
        for j in range(idxs[0].shape[0]):
            bbox = boxes[idxs[0][j], :] / im_scale
            #if int(labels[idxs[0][j]]) == 0:  # class0=people
            bboxs.append(bbox.tolist())
        bbox_all.append(bboxs)


    return heatmap_avg, bbox_all

def handle_heat(normal_heat, flipped_heat):
    """Compute the average of normal and flipped heatmap
    :param normal_heat: numpy array, the normal heatmap
    :param flipped_heat: numpy array, the flipped heatmap
    :returns: numpy arrays, the averaged heatmap
    """

    # The order to swap left and right of heatmap
    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                          13, 8, 9, 10, 15, 14, 17, 16))  # , 18

    averaged_heatmap = (normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

    return averaged_heatmap

def prn_network(joint_list,bboxs,file_name,image_id=0):

    in_thres = 0.21
    kps = joint_list

    bbox_list = bboxs

    idx = 0
    ks = []
    for j in range(18):  # joint type
        t = []
        for k in kps:
            if k[-1] == j:  # joint type
                x = k[0]
                y = k[1]
                v = k[2]
                if v > 0:
                    t.append([x, y, 1, idx])
                    idx += 1
        ks.append(t)
    peaks = ks

    w = 36
    h = 56

    bboxes = []
    for bbox_item in bbox_list:
        bboxes.append([bbox_item[0], bbox_item[1], bbox_item[2] - bbox_item[0], bbox_item[3] - bbox_item[1]])

    if len(bboxes) == 0 or len(peaks) == 0:
        prn_result = 0

    weights_bbox = np.zeros((len(bboxes), h, w, 4, 18))

    for joint_id, peak in enumerate(peaks):  # joint_id: which joint
        for instance_id, instance in enumerate(peak):  # instance_id: which people
            p_x = instance[0]
            p_y = instance[1]
            for bbox_id, b in enumerate(bboxes):
                is_inside = p_x > b[0] - b[2] * in_thres and \
                            p_y > b[1] - b[3] * in_thres and \
                            p_x < b[0] + b[2] * (1.0 + in_thres) and \
                            p_y < b[1] + b[3] * (1.0 + in_thres)
                if is_inside:
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])
                    x0 = int((p_x - b[0]) * x_scale)
                    y0 = int((p_y - b[1]) * y_scale)
                    if x0 >= w and y0 >= h:
                        x0 = w - 1
                        y0 = h - 1
                    elif x0 >= w:
                        x0 = w - 1
                    elif y0 >= h:
                        y0 = h - 1
                    elif x0 < 0 and y0 < 0:
                        x0 = 0
                        y0 = 0
                    elif x0 < 0:
                        x0 = 0
                    elif y0 < 0:
                        y0 = 0
                    p = 1e-9
                    weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]
    old_weights_bbox = np.copy(weights_bbox)


    idx_in_coco = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    prn_model = PRN_Seperate(56, 36, 1024)
    prn_model.load_weights("../Models/prn_epoch20_final.h5")

    for j in range(weights_bbox.shape[0]):
        for t in range(18):
            weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])

    output_bbox = []
    for j in range(weights_bbox.shape[0]):
        inp = weights_bbox[j, :, :, 0, :]
        # inp_idx_coco = inp[...,idx_in_coco]
        output = prn_model.predict(np.expand_dims(inp, axis=0))
        output_coco = np.copy(output[0])
        # output_coco = output_coco[...,[idx_in_coco.index(i) for i in range(18)]]
        output_bbox.append(output_coco)
    #     output = prn_model.predict(np.expand_dims(inp, axis=0))
    #     output_bbox.append(output[0])

    output_bbox = np.array(output_bbox)

    prn_result = []
    keypoints_score = []

    for t in range(18):
        indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
        keypoint = []
        for i in indexes:
            cr = crop(output_bbox[i[0], :, :, t], (i[1], i[2]), N=15)
            score = np.sum(cr)

            kp_id = old_weights_bbox[i[0], i[1], i[2], 2, t]
            kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
            p_score = old_weights_bbox[i[0], i[1], i[2], 3, t]  ## ??
            bbox_id = i[0]

            score = kp_score * score

            s = [kp_id, bbox_id, kp_score, score]
            keypoint.append(s)
        keypoints_score.append(keypoint)

    bbox_keypoints = np.zeros((weights_bbox.shape[0], 18, 3))
    bbox_ids = np.arange(len(bboxes)).tolist()

    # kp_id, bbox_id, kp_score, my_score
    for i in range(18):
        joint_keypoints = keypoints_score[i]
        if len(joint_keypoints) > 0:  # if have output result in one type keypoint

            kp_ids = list(set([x[0] for x in joint_keypoints]))

            table = np.zeros((len(bbox_ids), len(kp_ids), 4))

            for b_id, bbox in enumerate(bbox_ids):
                for k_id, kp in enumerate(kp_ids):
                    own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                    if len(own) > 0:
                        table[bbox, k_id] = own[0]
                    else:
                        table[bbox, k_id] = [0] * 4

            for b_id, bbox in enumerate(bbox_ids):  # all bbx, from 0 to ...

                row = np.argsort(-table[bbox, :, 3])  # in bbx(bbox), sort from big to small, keypoint score

                if table[bbox, row[0], 3] > 0:  # score
                    for r in row:  # all keypoints
                        if table[bbox, r, 3] > 0:
                            column = np.argsort(
                                -table[:, r, 3])  # sort all keypoints r, from big to small, bbx score

                            if bbox == column[0]:  # best bbx. best keypoint
                                bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][
                                    0]
                                break
                            else:  # for bbx column[0], the worst keypoint is row2[0],
                                row2 = np.argsort(table[column[0], :, 3])
                                if row2[0] == r:
                                    bbox_keypoints[bbox, i, :] = \
                                        [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                    break
        else:  # len(joint_keypoints) == 0:
            for j in range(weights_bbox.shape[0]):
                b = bboxes[j]
                x_scale = float(w) / math.ceil(b[2])
                y_scale = float(h) / math.ceil(b[3])

                for t in range(18):
                    indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                    if len(indexes) == 0:
                        max_index = np.argwhere(output_bbox[j, :, :, t] == np.max(output_bbox[j, :, :, t]))
                        bbox_keypoints[j, t, :] = [max_index[0][1] / x_scale + b[0],
                                                   max_index[0][0] / y_scale + b[1], 0]

    my_keypoints = []
    for i in range(bbox_keypoints.shape[0]):

        k = np.zeros(54)
        k[0::3] = bbox_keypoints[i, :, 0]
        k[1::3] = bbox_keypoints[i, :, 1]
        k[2::3] = bbox_keypoints[i, :, 2]

        pose_score = 0
        count = 0
        for f in range(18):
            if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                count += 1
            pose_score += bbox_keypoints[i, f, 2]
        pose_score /= 18.0

        my_keypoints.append(k)

        #     bbox_scaled = [178.64406,  92.25899, 262.08423, 440.77716], [ 98.51016,   88.719315, 189.15514,  439.4927  ] ,[268.5406, 99.68559, 355.15057, 430.58966]

        image_data = {
            'image_id': image_id,
            'file_name': file_name,
            'category_id': 1,
            'bbox': bboxes[i],
            'score': pose_score,
            'keypoints': k.tolist()
        }
        prn_result.append(image_data)

    return prn_result

def factor_closest(num, factor, is_ceil=True):
    """Returns the closest integer to `num` that is divisible by `factor`
    Actually, that's a lie. By default, we return the closest factor _greater_
    than the input. If, however, you set `it_ceil` to `False`, we return the
    closest factor _less_ than the input.
    """
    num = float(num) / factor
    num = np.ceil(num) if is_ceil else np.floor(num)
    return int(num) * factor

def crop_with_factor(im, dest_size, factor=32, pad_val=0, basedon='min'):
    """Scale and pad an image to the desired size and divisibility
    Scale the specified dimension of the input image to `dest_size` then pad
    the result until it is cleanly divisible by `factor`.
    Args:
        im (Image): The input image.
        dest_size (int): The desired size of the unpadded, scaled image's
            dimension specified by `basedon`.
        factor (int): Pad the scaled image until it is factorable
        pad_val (number): Value to pad with.
        basedon (string): Specifies which dimension to base the scaling on.
            Valid inputs are 'min', 'max', 'w', and 'h'. Defaults to 'min'.
    Returns:
        A tuple of three elements:
            - The scaled and padded image.
            - The scaling factor.
            - The size of the non-padded part of the resulting image.
    """
    # Compute the scaling factor.
    im_size_min, im_size_max = np.min(im.shape[0:2]), np.max(im.shape[0:2])
    im_base = {'min': im_size_min,
               'max': im_size_max,
               'w': im.shape[1],
               'h': im.shape[0]}
    im_scale = float(dest_size) / im_base.get(basedon, im_size_min)

    # Scale the image.
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    # Compute the padded image shape. Ensure it's divisible by factor.
    h, w = im.shape[:2]
    new_h, new_w = factor_closest(h, factor), factor_closest(w, factor)
    # new_ = max(new_h, new_w)
    new_shape = [new_h, new_w] if im.ndim < 3 else [new_h, new_w, im.shape[-1]]
    # new_shape = [new_, new_] if im.ndim < 3 else [new_, new_, im.shape[-1]]

    # Pad the image.
    im_padded = np.full(new_shape, fill_value=pad_val, dtype=im.dtype)
    im_padded[0:h, 0:w] = im

    return im_padded, im_scale, im.shape

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a MultiPose Network.')


    parser.add_argument('--coco_dir', help='Path to coco main dir')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    coco_eval(args.coco_dir)


if __name__ == '__main__':
    main()