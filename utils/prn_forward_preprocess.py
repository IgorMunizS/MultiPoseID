import keras.backend as K
import tensorflow as tf


def get_prn_input(inputs):
    heatmap = inputs[0]
    bbox = inputs[1]
    scores = inputs[2]

    heatmap = K.tf.Print(heatmap, [heatmap], "heatmap: ")
    indices = tf.where(K.greater(scores[0], 0.5))
    filtered_boxes = K.tf.gather_nd(bbox[0], indices)
    filtered_boxes = filtered_boxes / 4
    x1,y1,x2,y2 = filtered_boxes[:,0], filtered_boxes[:,1], filtered_boxes[:,2], filtered_boxes[:,3]
    filtered_boxes = tf.stack([y1,x1,y2,x2], axis=1) #tf.image.crop_and_resize bbox order

    filtered_boxes = filtered_boxes / 119

    filtered_boxes = K.tf.Print(filtered_boxes, [filtered_boxes], "filtered_boxes: ")
    shape_boxes = K.shape(filtered_boxes)
    heatmap_c = tf.image.crop_and_resize(heatmap, filtered_boxes, box_ind=tf.zeros(shape_boxes[0], tf.int32), crop_size=(56,36))

    heatmap_c = K.tf.Print(heatmap_c, [heatmap_c], "heatmap_cropped: ")
    # heatmap_c = K.tf.expand_dims(heatmap_c, axis=0)

    return heatmap_c


def get_prn_output_shape(inputs):

    return (None,56,36,18)

