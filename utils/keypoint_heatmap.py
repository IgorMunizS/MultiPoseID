import math
import numpy as np


def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
    """
    Creates stacked heatmaps for all joints + background. For 18 joints
    we would get an array height x width x 19.
    Size width and height should be the same as output from the network
    so this heatmap can be used to evaluate a loss function.
    :param num_maps: number of maps. for coco dataset we have 18 joints + 1 background
    :param height: height dimension of the network output
    :param width: width dimension of the network output
    :param all_joints: list of all joints (for coco: 18 items)
    :param sigma: parameter used to calculate a gaussian
    :param stride: parameter used to scale down the coordinates of joints. Those coords
            relate to the original image size
    :return: heat maps (height x width x num_maps)
    """
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    # heatmap[:, :, -1] = np.clip(np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0
