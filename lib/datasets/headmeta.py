from ..network.heads import (AssociationMeta, DetectionMeta,
                             IntensityMeta, ParsingMeta, OffsetMeta, CascadeMeta, CenterMeta)
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    KINEMATIC_TREE_SKELETON,
    MHP_CATEGORIES,
    MHP_KEYPOINTS,
    MHP_PERSON_SKELETON
)


def factory(head_names):
    if head_names is None:
        return None
    return [factory_single(hn) for hn in head_names]


def factory_single(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, COCO_CATEGORIES)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             COCO_KEYPOINTS,
                             COCO_PERSON_SIGMAS,
                             COCO_UPRIGHT_POSE,
                             COCO_PERSON_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               DENSER_COCO_PERSON_CONNECTIONS,
                               sparse_skeleton=COCO_PERSON_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               KINEMATIC_TREE_SKELETON)

    if head_name == 'pdf':
        return ParsingMeta(head_name)

    if head_name == 'offset':
        return OffsetMeta(head_name,
                          COCO_KEYPOINTS,
                          COCO_UPRIGHT_POSE)
    if head_name == 'cascade':
        return CascadeMeta(head_name,
                           COCO_KEYPOINTS,
                           COCO_UPRIGHT_POSE)

    if head_name == 'caf':
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               COCO_PERSON_SKELETON)
    if head_name == 'pcf':
        return CenterMeta(head_name)
    raise NotImplementedError
