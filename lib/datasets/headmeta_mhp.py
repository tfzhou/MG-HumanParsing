from ..network.heads import (AssociationMeta, DetectionMeta,
                             IntensityMeta, ParsingMeta, OffsetMeta,
                             CenterMeta, CascadeMeta)
from .constants import (
    MHP_KEYPOINTS,
    MHP_PERSON_SKELETON,
    MHP_PERSON_SIGMAS,
    MHP_UPRIGHT_POSE
)


def factory(head_names):
    if head_names is None:
        return None
    return [factory_single(hn) for hn in head_names]


def factory_single(head_name):
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             MHP_KEYPOINTS,
                             MHP_PERSON_SIGMAS,
                             MHP_UPRIGHT_POSE,
                             MHP_PERSON_SKELETON)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               MHP_KEYPOINTS,
                               MHP_PERSON_SIGMAS,
                               MHP_UPRIGHT_POSE,
                               MHP_PERSON_SKELETON)

    if head_name == 'pdf':
        return ParsingMeta(head_name)
    if head_name == 'offset':
        return OffsetMeta(head_name,
                          MHP_KEYPOINTS,
                          MHP_UPRIGHT_POSE)

    if head_name == 'caf':
        return AssociationMeta(head_name,
                               MHP_KEYPOINTS,
                               MHP_PERSON_SIGMAS,
                               MHP_UPRIGHT_POSE,
                               MHP_PERSON_SKELETON)

    if head_name == 'cascade':
        return CascadeMeta(head_name,
                           MHP_KEYPOINTS,
                           MHP_UPRIGHT_POSE)

    if head_name == 'pcf':
        return CenterMeta(head_name)

    raise NotImplementedError
