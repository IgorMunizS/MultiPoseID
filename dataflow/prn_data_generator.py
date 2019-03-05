import math
import numpy as np
from utils.prn_gaussian import gaussian, gaussian_multi_input_mp, gaussian_multi_output
from random import shuffle


def get_data(ann_data, coco, height, width,thres):
    weights = np.zeros((height, width, 18))
    output = np.zeros((height, width, 18))

    idx_in_coco = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    bbox = ann_data['bbox']
    x = int(bbox[0])
    y = int(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    x_scale = float(width) / math.ceil(w)
    y_scale = float(height) / math.ceil(h)

    kpx = ann_data['keypoints'][0::3]
    kpy = ann_data['keypoints'][1::3]
    kpv = ann_data['keypoints'][2::3]
    neck_x = (kpx[5] + kpx[6]) / 2
    neck_y = (kpy[5] + kpy[6]) / 2
    neck_v = min(kpv[5], kpv[6])
    kpx.append(neck_x)
    kpy.append(neck_y)
    kpv.append(neck_v)

    for j in range(18):
        idx = idx_in_coco[j]
        if kpv[idx] > 0:
            x0 = int((kpx[idx] - x) * x_scale)
            y0 = int((kpy[idx] - y) * y_scale)

            if x0 >= width and y0 >= height:
                output[height - 1, width - 1, j] = 1
            elif x0 >= width:
                output[y0, width - 1, j] = 1
            elif y0 >= height:
                output[height - 1, x0, j] = 1
            elif x0 < 0 and y0 < 0:
                output[0, 0, j] = 1
            elif x0 < 0:
                output[y0, 0, j] = 1
            elif y0 < 0:
                output[0, x0, j] = 1
            else:
                output[y0, x0, j] = 1

    # print(ann_data)
    img_id = ann_data['image_id']
    img_data = coco.loadImgs(img_id)[0]
    ann_data = coco.loadAnns(coco.getAnnIds(img_data['id']))
    # print(ann_data)
    for ann in ann_data:
        kpx = ann['keypoints'][0::3]
        kpy = ann['keypoints'][1::3]
        kpv = ann['keypoints'][2::3]
        neck_x = (kpx[5] + kpx[6]) / 2
        neck_y = (kpy[5] + kpy[6]) / 2
        neck_v = min(kpv[5], kpv[6])
        kpx.append(neck_x)
        kpy.append(neck_y)
        kpv.append(neck_v)

        for j in range(18):
            idx = idx_in_coco[j]
            if kpv[idx] > 0:
                if (kpx[idx] > bbox[0] - bbox[2] * thres and kpx[idx] < bbox[0] + bbox[2] * (1 + thres)):
                    if (kpy[idx] > bbox[1] - bbox[3] * thres and kpy[idx] < bbox[1] + bbox[3] * (1 + thres)):
                        x0 = int((kpx[idx] - x) * x_scale)
                        y0 = int((kpy[idx] - y) * y_scale)

                        if x0 >= width and y0 >= height:
                            weights[height - 1, width - 1, j] = 1
                        elif x0 >= width:
                            weights[y0, width - 1, j] = 1
                        elif y0 >= height:
                            weights[height - 1, x0, j] = 1
                        elif x0 < 0 and y0 < 0:
                            weights[0, 0, j] = 1
                        elif x0 < 0:
                            weights[y0, 0, j] = 1
                        elif y0 < 0:
                            weights[0, x0, j] = 1
                        else:
                            weights[y0, x0, j] = 1

    for t in range(18):
        weights[:, :, t] = gaussian(weights[:, :, t])
    output  =  gaussian(output, sigma=2, mode='constant', multichannel=True)
    #weights = gaussian_multi_input_mp(weights)
    return weights, output, ann_data


def get_anns(coco):
    '''
    :param coco: COCO instance
    :return: anns: List of annotations that contain person with at least 6 keypoints
    '''
    ann_ids = coco.getAnnIds()
    anns = []
    for i in ann_ids:
        ann = coco.loadAnns(i)[0]
        if ann['iscrowd'] == 0 and ann['num_keypoints'] > 3:
            anns.append(ann) # ann
    sorted_list = sorted(anns, key=lambda k: k['num_keypoints'], reverse=True)
    return sorted_list


def train_bbox_generator(coco_train,batch_size,height,width,thres):
    anns = get_anns(coco_train)
    while 1:
        shuffle(anns)
        for i in range(0, len(anns) // batch_size, batch_size):
            X = np.zeros((batch_size, height, width, 18))
            Y = np.zeros((batch_size, height, width, 18))
            for j in range(batch_size):
                ann_data = anns[i+j]
                try:
                    x, y, z = get_data(ann_data, coco_train, height, width, thres)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y




def val_bbox_generator(coco_val, batch_size,height,width,thres):
    ann_ids = coco_val.getAnnIds()
    while 1:
        shuffle(ann_ids)
        for i in range(len(ann_ids) // batch_size):
            X = np.zeros((batch_size, height, width, 18))
            Y = np.zeros((batch_size, height, width, 18))
            for j in range(batch_size):
                ann_data = coco_val.loadAnns(ann_ids[i + j])[0]
                try:
                    x, y, _ = get_data(ann_data, coco_val,height,width,thres)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y
