import cv2
import math
import numpy as np


def rotate_box(bbox, width, height, angle_degrees):
    """Input bounding box is of the form x, y, width, height."""

    cangle = math.cos(angle_degrees / 180.0 * math.pi)
    sangle = math.sin(angle_degrees / 180.0 * math.pi)

    four_corners = np.array([
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0], bbox[1] + bbox[3]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
    ])

    x_old = four_corners[:, 0].copy() - width/2
    y_old = four_corners[:, 1].copy() - height/2
    four_corners[:, 0] = width/2 + cangle * x_old + sangle * y_old
    four_corners[:, 1] = height/2 - sangle * x_old + cangle * y_old

    x = np.min(four_corners[:, 0])
    y = np.min(four_corners[:, 1])
    xmax = np.max(four_corners[:, 0])
    ymax = np.max(four_corners[:, 1])

    return np.array([x, y, xmax - x, ymax - y])


def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge
