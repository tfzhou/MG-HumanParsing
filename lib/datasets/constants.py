import numpy as np

################### MHP #######################

MHP_CATEGORIES = [
    "Cap/Hat",
    "Helmet",
    "Face",
    "Hair",
    "Left-arm",
    "Right-arm",
    "Left-hand",
    "Right-hand",
    "Protector",
    "Bikini/bra",
    "Jacket/windbreaker/hoodie",
    "Tee-shirt",
    "Polo-shirt",
    "Sweater",
    "Singlet",
    "Torso-skin",
    "Pants",
    "Shorts/swim-shorts",
    "Skirt",
    "Stockings",
    "Socks",
    "Left-boot",
    "Right-boot",
    "Left-shoe",
    "Right-shoe",
    "Left-highheel",
    "Right-highheel",
    "Left-sandal",
    "Right-sandal",
    "Left-leg",
    "Right-leg",
    "Left-foot",
    "Right-foot",
    "Coat",
    "Dress",
    "Robe",
    "Jumpsuit",
    "Other-full-body-clothes",
    "Headwear",
    "Backpack",
    "Ball",
    "Bats",
    "Belt",
    "Bottle",
    "Carrybag",
    "Cases",
    "Sunglasses",
    "Eyewear",
    "Glove",
    "Scarf",
    "Umbrella",
    "Wallet/purse",
    "Watch",
    "Wristband",
    "Tie",
    "other-accessaries",
    "other-upper-body-clothes",
    "other-lower-body-clothes"
]

MHP_HFLIP = {
    "Left-arm": "Right-arm",
    "Right-arm": "Left-arm",
    "Left-hand": "Right-hand",
    "Right-hand": "Left-hand",
    "Left-boot": "Right-boot",
    "Right-boot": "Left-boot",
    "Left-shoe": "Right-shoe",
    "Right-shoe": "Left-shoe",
    "Left-highheel": "Right-highheel",
    "Right-highheel": "Left-highheel",
    "Left-sandal": "Right-sandal",
    "Right-sandal": "Left-sandal",
    "Left-leg": "Right-leg",
    "Right-leg": "Left-leg",
    "Left-foot": "Right-foot",
    "Right-foot": "Left-foot"
}

MHP_KEYPOINTS = ["Right-ankle", "Right-knee", "Right-hip", "Left-hip",
                 "Left-knee", "Left-ankle", "Pelvis", "Thorax", "Upper-neck",
                 "Head-top", "Right-wrist", "Right-elbow", "Right-shoulder",
                 "Left-shoulder", "Left-elbow", "Left-wrist"]

MHP_KEYPOINTS_HFLIP = {
    "Right-ankle": "Left-ankle",
    "Right-knee": "Left-knee",
    "Right-hip": "Left-hip",
    "Left-hip": "Right-hip",
    "Left-knee": "Right-knee",
    "Left-ankle": "Right-ankle",
    "Pelvis": "Pelvis",
    "Thorax": "Thorax",
    "Upper-neck": "Upper-neck",
    "Head-top": "Head-top",
    "Right-wrist": "Left-wrist",
    "Right-elbow": "Left-elbow",
    "Right-shoulder": "Left-shoulder",
    "Left-shoulder": "Right-shoulder",
    "Left-elbow": "Right-elbow",
    "Left-wrist": "Right-wrist"
}

MHP_PERSON_SIGMAS = [
    0.089,  # Right-ankle
    0.087,  # Right-knee
    0.107,  # Right-hip
    0.107,  # Left-hip
    0.087,  # Left-knee
    0.089,  # Left-ankle
    0.087,  # Pelvis
    0.107,  # Thorax
    0.079,  # Upper-neck
    0.026,  # Head-top
    0.062,  # Right-wrist
    0.072,  # Right-elbow
    0.079,  # Right-shoulder
    0.079,  # Left-shoulder
    0.072,  # Left-elbow
    0.062,  # Left-wrist
]

MHP_UPRIGHT_POSE = np.array([
    [1.4, 0.1, 2.0],    # 'right_ankle',     # 1
    [1.4, 2.1, 2.0],    # 'right_knee',      # 2
    [1.26, 4.0, 2.0],   # 'right_hip',       # 3
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 4
    [-1.4, 2.0, 2.0],   # 'left_knee',       # 4
    [-1.4, 0.0, 2.0],   # 'left_ankle',      # 6
    [0.0, 4.0, 2.0],    # 'pelvis',          # 7
    [0.0, 6.5, 2.0],      # Thorax,   # 8
    [0.0, 9.0, 2.0],      # Upper-neck  # 9
    [0.0, 10.0, 2.0],      # Head-top  # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
])

MHP_PERSON_SKELETON = [[1, 2], [2, 3], [3, 7], [4, 7], [4, 5],
                       [5, 6], [7, 8], [8, 9], [9,  10], [9, 13], [9, 14],
                       [11, 12], [12, 13], [14, 15], [15, 16],
                       [13, 3], [14, 4], [13, 14]]

################### DensePose & COCO #######################

DENSEPOSE_CATEGORIES = [
    "Torso",
    "Right-Hand",
    "Left-Hand",
    "Left-Foot",
    "Right-Foot",
    "UpperRight-Leg",
    "UpperLeft-Leg",
    "LowerRight-Leg",
    "LowerLeft-Leg",
    "UpperLeft-Arm",
    "UpperRight-Arm",
    "LowerLeft-Arm",
    "LowerRight-Arm",
    "Head"
]


DENSEPOSE_HFLIP = {
    'Torso': 'Torso',
    'Right-Hand': 'Left-Hand',
    'Left-Hand': 'Right-Hand',
    'Left-Foot': 'Right-Foot',
    'Right-Foot': 'Left-Foot',
    'UpperRight-Leg': 'UpperLeft-Leg',
    'UpperLeft-Leg': 'UpperRight-Leg',
    'LowerRight-Leg': 'LowerLeft-Leg',
    'LowerLeft-Leg': 'LowerRight-Leg',
    'UpperLeft-Arm': 'UpperRight-Arm',
    'UpperRight-Arm': 'UpperLeft-Arm',
    'LowerLeft-Arm': 'LowerRight-Arm',
    'LowerRight-Arm': 'LowerLeft-Arm',
    'Head': 'Head'
}


COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]


KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]


COCO_KEYPOINTS = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
]


COCO_UPRIGHT_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


COCO_DAVINCI_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-3.3, 9.0, 2.0],  # 'left_elbow',      # 8
    [3.3, 9.2, 2.0],  # 'right_elbow',     # 9
    [-4.5, 10.5, 2.0],  # 'left_wrist',      # 10
    [4.5, 10.7, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-2.0, 2.0, 2.0],  # 'left_knee',       # 14
    [2.0, 2.1, 2.0],  # 'right_knee',      # 15
    [-2.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [2.4, 0.1, 2.0],  # 'right_ankle',     # 17
])


HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


DENSER_COCO_PERSON_SKELETON = [
    (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
    (1, 6), (1, 7), (2, 6), (3, 7),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
    (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
    (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
    (8, 9), (10, 11),
    (10, 12), (11, 13),
    (10, 14), (11, 15),
    (14, 12), (15, 13), (12, 15), (13, 14),
    (12, 16), (13, 17),
    (16, 14), (17, 15), (14, 17), (15, 16),
    (14, 15), (16, 17),
]


DENSER_COCO_PERSON_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
]


COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose):
    from ..annotation import Annotation  # pylint: disable=import-outside-toplevel
    from .. import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)

    ann = Annotation(keypoints=COCO_KEYPOINTS, skeleton=COCO_PERSON_SKELETON)
    ann.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    draw_ann(ann, filename='docs/skeleton_coco.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=COCO_KEYPOINTS, skeleton=KINEMATIC_TREE_SKELETON)
    ann.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    draw_ann(ann, filename='docs/skeleton_kinematic_tree.png', keypoint_painter=keypoint_painter)

    ann = Annotation(keypoints=COCO_KEYPOINTS, skeleton=DENSER_COCO_PERSON_SKELETON)
    ann.set(pose, np.array(COCO_PERSON_SIGMAS) * scale)
    draw_ann(ann, filename='docs/skeleton_dense.png', keypoint_painter=keypoint_painter)


def draw_mhp_skeletons(pose):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)

    ann = Annotation(keypoints=MHP_KEYPOINTS, skeleton=MHP_PERSON_SKELETON)
    ann.set(pose, np.array(MHP_PERSON_SIGMAS) * scale)
    draw_ann(ann, filename='skeleton_mhp.png', keypoint_painter=keypoint_painter)


def print_associations():
    for j1, j2 in COCO_PERSON_SKELETON:
        print(COCO_KEYPOINTS[j1 - 1], '-', COCO_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    #print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    #draw_skeletons(COCO_UPRIGHT_POSE)

    print(np.sum(COCO_PERSON_SIGMAS))
    print(np.sum(MHP_PERSON_SIGMAS))

    draw_mhp_skeletons(MHP_UPRIGHT_POSE)


