import logging
import os
import torch

import torchvision

from . import basenetworks, heads, nets
from .. import datasets

from . import resnet, xception, hrnet
from .decoder import (SinglePanopticDeepLabHead, SinglePanopticDeepLabHeadFused,
                      SinglePanopticDeepLabDecoder, CascadeRefinementHead, CascadeRefinementHeadXception)

# generate hash values with: shasum -a 256 filename.pkl


CHECKPOINT_URLS = {
    'resnet50': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                 'v0.11.2/resnet50-200527-171310-cif-caf-caf25-o10s-c0b7ae80.pkl'),
    'shufflenetv2k16w': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                         'v0.11.0/shufflenetv2k16w-200510-221334-cif-caf-caf25-o10s-604c5956.pkl'),
    'shufflenetv2k30w': ('http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
                         'v0.11.0/shufflenetv2k30w-200510-104256-cif-caf-caf25-o10s-0b5ba06f.pkl'),
}

LOG = logging.getLogger(__name__)


def factory_from_args(args):
    return factory(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        base_name=args.basenet,
        head_names=args.headnets,
        pretrained=args.pretrained,
        dense_connections=getattr(args, 'dense_connections', False),
        cross_talk=args.cross_talk,
        two_scale=args.two_scale,
        multi_scale=args.multi_scale,
        multi_scale_hflip=args.multi_scale_hflip,
        download_progress=args.download_progress,
        num_classes=args.num_classes,
        with_edge=args.with_edge
    )


def local_checkpoint_path(checkpoint):
    if os.path.exists(checkpoint):
        return checkpoint

    if checkpoint in CHECKPOINT_URLS:
        url = CHECKPOINT_URLS[checkpoint]

        file_name = os.path.join(
            os.getenv('XDG_CACHE_HOME', os.path.join(os.getenv('HOME'), '.cache')),
            'torch',
            'checkpoints',
            os.path.basename(url),
        )
        print(file_name, url, os.path.basename(url))

        if os.path.exists(file_name):
            return file_name

    return None


# pylint: disable=too-many-branches,too-many-statements
def factory(
        *,
        dataset='cocokp',
        checkpoint=None,
        base_name=None,
        head_names=None,
        pretrained=True,
        dense_connections=False,
        cross_talk=0.0,
        two_scale=False,
        multi_scale=False,
        multi_scale_hflip=True,
        download_progress=True,
        num_classes=15,
        with_edge=False):

    if base_name:
        assert head_names
        assert checkpoint is None
        net_cpu = factory_from_scratch(base_name, head_names,
                                       num_classes=num_classes,
                                       dataset=dataset,
                                       pretrained=pretrained,
                                       with_edge=with_edge)
        epoch = 0
    else:
        assert base_name is None
        assert head_names is None

        if not checkpoint:
            checkpoint = 'shufflenetv2k16w'

        if checkpoint == 'resnet18':
            raise Exception('this pretrained model is currently not available')
        if checkpoint == 'resnet101':
            raise Exception('this pretrained model is currently not available')
        checkpoint = CHECKPOINT_URLS.get(checkpoint, checkpoint)

        if checkpoint.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint,
                check_hash=not checkpoint.startswith('https'),
                progress=download_progress)
        else:
            checkpoint = torch.load(checkpoint)

        net_cpu = checkpoint['model']
        epoch = checkpoint['epoch']

        # normalize for backwards compatibility
        nets.model_migration(net_cpu)

        # initialize for eval
        net_cpu.eval()

    cif_indices = [0]
    caf_indices = [1]
    if not any(isinstance(h.meta, heads.AssociationMeta) for h in net_cpu.head_nets):
        caf_indices = []
    if dense_connections and not multi_scale:
        caf_indices = [1, 2]
    elif dense_connections and multi_scale:
        cif_indices = [v * 3 + 1 for v in range(10)]
        caf_indices = [v * 3 + 2 for v in range(10)]
    if isinstance(net_cpu.head_nets[0].meta, heads.DetectionMeta):
        net_cpu.process_heads = heads.CifdetCollector(cif_indices)
    else:
        net_cpu.process_heads = heads.CifCafCollector(cif_indices, caf_indices)
    net_cpu.cross_talk = cross_talk

    if two_scale:
        net_cpu = nets.Shell2Scale(net_cpu.base_net, net_cpu.head_nets)

    if multi_scale:
        net_cpu = nets.ShellMultiScale(net_cpu.base_net, net_cpu.head_nets,
                                       process_heads=net_cpu.process_heads,
                                       include_hflip=multi_scale_hflip)

    return net_cpu, epoch


def factory_from_scratch(basename, head_names, num_classes, *, dataset='cocokp',
                         pretrained=True, with_edge=False):
    if dataset == 'mhp':
        head_metas = datasets.headmeta_mhp.factory(head_names)
    else:
        head_metas = datasets.headmeta.factory(head_names)

    if 'resnet50' in basename:
        return resnet_factory_from_scratch(basename, 2048, num_classes, head_metas,
                                           pretrained=pretrained, with_edge=with_edge)
    if 'hrnet' in basename:
        return hrnet_fractory_from_scratch(basename, 384, num_classes, head_metas,
                                           pretrained=pretrained, with_edge=with_edge)
    if 'xception' in basename:
        return xception_factory_from_scratch(basename, 2048, num_classes, head_metas, with_edge=with_edge)

    raise Exception('unknown base network in {}'.format(basename))


def generic_factory_from_scratch(basename, base_vision, out_features, head_metas):
    basenet = basenetworks.BaseNetwork(
        base_vision,
        basename,
        stride=16,
        out_features=out_features,
    )

    headnets = [heads.CompositeFieldFused(h, basenet.out_features) for h in head_metas]

    net_cpu = nets.Shell(basenet, headnets)
    nets.model_defaults(net_cpu)
    LOG.debug(net_cpu)
    return net_cpu


def hrnet_fractory_from_scratch(basename, out_features, num_classes, head_metas, *,
                                pretrained=True, with_edge=True):
    backbone = hrnet.__dict__[basename](pretrained=pretrained)

    output_stride = 16
    basenet = basenetworks.BaseNetwork(
        backbone,
        basename,
        stride=output_stride,
        out_features=out_features)

    # ============== define decoder and head networks ==================
    pose_heads = []
    segm_heads = []
    center_decoder = None
    center_head = None
    cascade_head = None

    segm_num_classes = [num_classes]
    segm_class_keys = ['semantic']
    if with_edge:
        segm_num_classes += [1]
        segm_class_keys += ['edge']

    cfg = {
        # pose cfg
        'pose_decoder_channels': 256,

        # segm cfg
        'segm_decoder_channels': 256,
        'segm_head_channels': 256,
        'segm_num_classes': segm_num_classes,
        'segm_class_keys': segm_class_keys,

        # offset cfg
        'offset_decoder_channels': 256,
        'offset_head_channels': 32,

        # center cfg
        'center_decoder_channels': 256,
        'center_head_channels': 32
    }

    pose_decoder = SinglePanopticDeepLabDecoder(
        in_channels=basenet.out_features,
        feature_key='res5',
        low_level_channels=(192, 96, 48),
        low_level_key=["res4", "res3", "res2"],
        low_level_channels_project=(128, 64, 32),
        decoder_channels=cfg['pose_decoder_channels'],
        atrous_rates=(3, 6, 9),
        aspp_channels=256)

    for head in head_metas:
        if head.name == 'cascade':
            cascade_head = CascadeRefinementHead(
                meta=head,
                num_classes=[num_classes, 2],
                class_key=['semantic', 'offset'],
                head_channels=256)
        elif head.name == 'pcf':
            center_decoder = SinglePanopticDeepLabDecoder(
                in_channels=basenet.out_features,
                feature_key='res5',
                low_level_channels=(192, 96, 48),
                low_level_key=["res4", "res3", "res2"],
                low_level_channels_project=(128, 64, 32),
                decoder_channels=cfg['center_decoder_channels'],
                atrous_rates=(3, 6, 9),
                aspp_channels=256)
            center_head = SinglePanopticDeepLabHead(
                head,
                decoder_channels=256,
                head_channels=cfg['center_decoder_channels'],
                num_classes=[1, 2],
                class_key=['center', 'center_offset'])
        else:
            pose_heads.append(heads.CompositeFieldFused(
                head, in_features=cfg['pose_decoder_channels']))

    net_cpu = nets.ShellHRNet(basenet,
                              pose_decoder=pose_decoder, pose_heads=pose_heads,
                              cascade_head=cascade_head,
                              center_decoder=center_decoder,
                              center_head=center_head,
                              with_edge=with_edge)

    #for head in head_metas:
    #    if head.name == 'pdf':
    #        segm_decoder = SinglePanopticDeepLabDecoder(
    #            in_channels=basenet.out_features,
    #            feature_key='res5',
    #            low_level_channels=(1024, 512),
    #            low_level_key=["res4", "res3"],
    #            low_level_channels_project=(128, 64),
    #            decoder_channels=cfg['segm_decoder_channels'],
    #            atrous_rates=(3, 6, 9),
    #            aspp_channels=256)

    #        segm_heads = SinglePanopticDeepLabHead(
    #            head,
    #            decoder_channels=cfg['segm_decoder_channels'],
    #            head_channels=cfg['segm_head_channels'],
    #            num_classes=cfg['segm_num_classes'],
    #            class_key=cfg['segm_class_keys'])
    #    elif head.name == 'offset':
    #        offset_decoder = SinglePanopticDeepLabDecoder(
    #            in_channels=basenet.out_features,
    #            feature_key='res5',
    #            low_level_channels=(1024, 512),
    #            low_level_key=["res4", "res3"],
    #            low_level_channels_project=(128, 64),
    #            decoder_channels=cfg['offset_decoder_channels'],
    #            atrous_rates=(3, 6, 9),
    #            aspp_channels=256)
    #        decoder_channel = cfg['offset_decoder_channels'] + \
    #                          cfg['segm_decoder_channels'] + \
    #                          cfg['pose_decoder_channels']
    #        offset_heads = SinglePanopticDeepLabHead(
    #            head,
    #            decoder_channels=decoder_channel,
    #            head_channels=cfg['offset_decoder_channels'],
    #            num_classes=[2],
    #            class_key=['offset'])
    #    else:
    #        pose_heads.append(heads.CompositeFieldFused(
    #            head, in_features=cfg['pose_decoder_channels']))

    # ================== define network =====================
    #net_cpu = nets.Shell(basenet,
    #                     pose_decoder=pose_decoder, pose_heads=pose_heads,
    #                     segm_decoder=segm_decoder, segm_heads=segm_heads,
    #                     offset_decoder=offset_decoder, offset_heads=offset_heads,
    #                     with_edge=with_edge)
    print(net_cpu)
    nets.model_defaults(net_cpu)
    return net_cpu


#def xception_factory_from_scratch(basename, out_features, num_classes, head_metas, *, pretrained=True):
#    backbone = xception.__dict__['xception65'](
#        pretrained=pretrained,
#        replace_stride_with_dilation=(False, False, True),
#    )
#
#    output_stride = 16
#    basenet = basenetworks.BaseNetwork(
#        backbone,
#        basename,
#        stride=output_stride,
#        out_features=out_features,
#    )
#
#    # ============== define decoder and head networks ==================
#    pose_heads = []
#    segm_heads = []
#    center_decoder = None
#    center_head = None
#    cascade_head = None
#
#    segm_num_classes = [num_classes]
#    segm_class_keys = ['semantic']
#
#    cfg = {
#        # pose cfg
#        'pose_decoder_channels': 256,
#
#        # segm cfg
#        'segm_decoder_channels': 256,
#        'segm_head_channels': 256,
#        'segm_num_classes': segm_num_classes,
#        'segm_class_keys': segm_class_keys,
#
#        # offset cfg
#        'offset_decoder_channels': 256,
#        'offset_head_channels': 32,
#
#        # center cfg
#        'center_decoder_channels': 256,
#        'center_head_channels': 32
#    }
#
#    pose_decoder = SinglePanopticDeepLabDecoder(
#        in_channels=basenet.out_features,
#        feature_key='res5',
#        low_level_channels=(728, 728, 256),
#        low_level_key=["res4", "res3", "res2"],
#        low_level_channels_project=(128, 64, 32),
#        decoder_channels=cfg['pose_decoder_channels'],
#        atrous_rates=(3, 6, 9),
#        aspp_channels=256)
#
#    for head in head_metas:
#        if head.name == 'cascade':
#            cascade_head = CascadeRefinementHeadXception(
#                meta=head,
#                num_classes=[num_classes, 2],
#                class_key=['semantic', 'offset'],
#                head_channels=256)
#        elif head.name == 'pcf':
#            center_decoder = SinglePanopticDeepLabDecoder(
#                in_channels=basenet.out_features,
#                feature_key='res5',
#                low_level_channels=(728, 728, 256),
#                low_level_key=["res4", "res3", "res2"],
#                low_level_channels_project=(128, 64, 32),
#                decoder_channels=cfg['center_decoder_channels'],
#                atrous_rates=(3, 6, 9),
#                aspp_channels=256)
#            center_head = SinglePanopticDeepLabHead(
#                head,
#                decoder_channels=256,
#                head_channels=cfg['center_decoder_channels'],
#                num_classes=[1, 2],
#                class_key=['center', 'center_offset'])
#        else:
#            pose_heads.append(heads.CompositeFieldFused(
#                head, in_features=cfg['pose_decoder_channels']))
#
#    net_cpu = nets.ShellHRNet(basenet,
#                              pose_decoder=pose_decoder, pose_heads=pose_heads,
#                              cascade_head=cascade_head,
#                              center_decoder=center_decoder,
#                              center_head=center_head)
#
#    nets.model_defaults(net_cpu)
#    return net_cpu


def xception_factory_from_scratch(basename, out_features, num_classes, head_metas, *, pretrained=True, with_edge=False):
    backbone = xception.__dict__['xception65'](
        pretrained=pretrained,
        replace_stride_with_dilation=(False, False, True),
    )

    output_stride = 16
    basenet = basenetworks.BaseNetwork(
        backbone,
        basename,
        stride=output_stride,
        out_features=out_features,
    )

    # ============== define decoder and head networks ==================
    pose_heads = []
    segm_heads = []
    center_decoder = None
    center_head = None
    cascade_head = None

    segm_num_classes = [num_classes]
    segm_class_keys = ['semantic']
    if with_edge:
        segm_num_classes += [1]
        segm_class_keys += ['edge']

    cfg = {
        # pose cfg
        'pose_decoder_channels': 256,

        # segm cfg
        'segm_decoder_channels': 256,
        'segm_head_channels': 256,
        'segm_num_classes': segm_num_classes,
        'segm_class_keys': segm_class_keys,

        # offset cfg
        'offset_decoder_channels': 256,
        'offset_head_channels': 32,

        # center cfg
        'center_decoder_channels': 256,
        'center_head_channels': 32
    }

    pose_decoder = SinglePanopticDeepLabDecoder(
        in_channels=basenet.out_features,
        feature_key='res5',
        low_level_channels=(728, 728, 256),
        low_level_key=["res4", "res3", "res2"],
        low_level_channels_project=(128, 64, 32),
        decoder_channels=cfg['pose_decoder_channels'],
        atrous_rates=(3, 6, 9),
        aspp_channels=256)

    for head in head_metas:
        if head.name == 'cascade':
            cascade_head = CascadeRefinementHeadXception(
                meta=head,
                num_classes=[2],
                class_key=['offset'],
                head_channels=256)
        elif head.name == 'pdf':
            semantic_decoder = SinglePanopticDeepLabDecoder(
                in_channels=basenet.out_features,
                feature_key='res5',
                low_level_channels=(728, 728, 256),
                low_level_key=["res4", "res3", "res2"],
                low_level_channels_project=(128, 64, 32),
                decoder_channels=cfg['segm_decoder_channels'],
                atrous_rates=(3, 6, 9),
                aspp_channels=256)
            semantic_head = SinglePanopticDeepLabHead(
                head,
                decoder_channels=256,
                head_channels=cfg['segm_decoder_channels'],
                num_classes=segm_num_classes,
                class_key=segm_class_keys
            )
        #elif head.name == 'offset':
        #    offset_decoder = SinglePanopticDeepLabDecoder(
        #        in_channels=basenet.out_features,
        #        feature_key='res5',
        #        low_level_channels=(728, 728, 256),
        #        low_level_key=["res4", "res3", "res2"],
        #        low_level_channels_project=(128, 64, 32),
        #        decoder_channels=cfg['offset_decoder_channels'],
        #        atrous_rates=(3, 6, 9),
        #        aspp_channels=256)
        #    offset_head = SinglePanopticDeepLabHead(
        #        head,
        #        decoder_channels=256,
        #        head_channels=cfg['offset_decoder_channels'],
        #        num_classes=[2],
        #        class_key=['offset']
        #    )
        elif head.name == 'pcf':
            center_decoder = SinglePanopticDeepLabDecoder(
                in_channels=basenet.out_features,
                feature_key='res5',
                low_level_channels=(728, 728, 256),
                low_level_key=["res4", "res3", "res2"],
                low_level_channels_project=(128, 64, 32),
                decoder_channels=cfg['center_decoder_channels'],
                atrous_rates=(3, 6, 9),
                aspp_channels=256)
            center_head = SinglePanopticDeepLabHead(
                head,
                decoder_channels=256,
                head_channels=cfg['center_decoder_channels'],
                num_classes=[1, 2],
                class_key=['center', 'center_offset'])
        else:
            pose_heads.append(heads.CompositeFieldFused(
                head, in_features=cfg['pose_decoder_channels']))

    net_cpu = nets.ShellXception(basenet,
                                 semantic_decoder=semantic_decoder, semantic_head=semantic_head,
                                 cascade_head=cascade_head,
                                 #offset_decoder=offset_decoder, offset_head=offset_head,
                              pose_decoder=pose_decoder, pose_heads=pose_heads,
                              center_decoder=center_decoder,
                              center_head=center_head)

    nets.model_defaults(net_cpu)
    return net_cpu


def resnet_factory_from_scratch(basename, out_features, num_classes,
                                head_metas, *,
                                pretrained=True, with_edge=False):
    use_pool = True if 'pool' in basename else False

    backbone = resnet.__dict__['resnet50'](
        pretrained=pretrained,
        replace_stride_with_dilation=(False, False, False),
        use_pool=use_pool
    )

    output_stride = 16
    basenet = basenetworks.BaseNetwork(
        backbone,
        basename,
        stride=output_stride,
        out_features=out_features,
    )

    # ============== define decoder and head networks ==================
    pose_heads = []
    segm_heads = []
    #segm_decoder, segm_heads, offset_decoder, offset_heads = None, None, None, None

    segm_num_classes = [num_classes]
    segm_class_keys = ['semantic']
    if with_edge:
        segm_num_classes += [1]
        segm_class_keys += ['edge']

    cfg = {
        # pose cfg
        'pose_decoder_channels': 256,

        # segm cfg
        'segm_decoder_channels': 256,
        'segm_head_channels': 256,
        'segm_num_classes': segm_num_classes,
        'segm_class_keys': segm_class_keys,

        # offset cfg
        'offset_decoder_channels': 256,
        'offset_head_channels': 32
    }

    pose_decoder = SinglePanopticDeepLabDecoder(
        in_channels=basenet.out_features,
        feature_key='res5',
        low_level_channels=(1024, 512),
        low_level_key=["res4", "res3"],
        low_level_channels_project=(128, 64),
        decoder_channels=cfg['pose_decoder_channels'],
        atrous_rates=(3, 6, 9),
        aspp_channels=256)

    segm_decoder = SinglePanopticDeepLabDecoder(
        in_channels=basenet.out_features,
        feature_key='res5',
        low_level_channels=(1024, 512),
        low_level_key=["res4", "res3"],
        low_level_channels_project=(128, 64),
        decoder_channels=cfg['segm_decoder_channels'],
        atrous_rates=(3, 6, 9),
        aspp_channels=256)

    for head in head_metas:
        if head.name == 'pdf':
            segm_head = SinglePanopticDeepLabHeadFused(
                head,
                decoder_channels=cfg['segm_decoder_channels'],
                head_channels=cfg['segm_head_channels'],
                num_classes=cfg['segm_num_classes'],
                class_key=cfg['segm_class_keys'])
            segm_heads.append(segm_head)
        elif head.name == 'offset':
            offset_head = SinglePanopticDeepLabHeadFused(
                head,
                decoder_channels=cfg['offset_decoder_channels'],
                head_channels=cfg['offset_decoder_channels'],
                num_classes=[2],
                class_key=['offset'])
            segm_heads.append(offset_head)
        else:
            pose_heads.append(heads.CompositeFieldFused(
                head, in_features=cfg['pose_decoder_channels'] + 14 + 1 + 1))

    net_cpu = nets.Shell(basenet,
                         pose_decoder=pose_decoder, pose_heads=pose_heads,
                         segm_decoder=segm_decoder, segm_heads=segm_heads,
                         with_edge=with_edge)

    #for head in head_metas:
    #    if head.name == 'pdf':
    #        segm_decoder = SinglePanopticDeepLabDecoder(
    #            in_channels=basenet.out_features,
    #            feature_key='res5',
    #            low_level_channels=(1024, 512),
    #            low_level_key=["res4", "res3"],
    #            low_level_channels_project=(128, 64),
    #            decoder_channels=cfg['segm_decoder_channels'],
    #            atrous_rates=(3, 6, 9),
    #            aspp_channels=256)

    #        segm_heads = SinglePanopticDeepLabHead(
    #            head,
    #            decoder_channels=cfg['segm_decoder_channels'],
    #            head_channels=cfg['segm_head_channels'],
    #            num_classes=cfg['segm_num_classes'],
    #            class_key=cfg['segm_class_keys'])
    #    elif head.name == 'offset':
    #        offset_decoder = SinglePanopticDeepLabDecoder(
    #            in_channels=basenet.out_features,
    #            feature_key='res5',
    #            low_level_channels=(1024, 512),
    #            low_level_key=["res4", "res3"],
    #            low_level_channels_project=(128, 64),
    #            decoder_channels=cfg['offset_decoder_channels'],
    #            atrous_rates=(3, 6, 9),
    #            aspp_channels=256)
    #        decoder_channel = cfg['offset_decoder_channels'] + \
    #                          cfg['segm_decoder_channels'] + \
    #                          cfg['pose_decoder_channels']
    #        offset_heads = SinglePanopticDeepLabHead(
    #            head,
    #            decoder_channels=decoder_channel,
    #            head_channels=cfg['offset_decoder_channels'],
    #            num_classes=[2],
    #            class_key=['offset'])
    #    else:
    #        pose_heads.append(heads.CompositeFieldFused(
    #            head, in_features=cfg['pose_decoder_channels']))

    # ================== define network =====================
    #net_cpu = nets.Shell(basenet,
    #                     pose_decoder=pose_decoder, pose_heads=pose_heads,
    #                     segm_decoder=segm_decoder, segm_heads=segm_heads,
    #                     offset_decoder=offset_decoder, offset_heads=offset_heads,
    #                     with_edge=with_edge)
    print(net_cpu)
    nets.model_defaults(net_cpu)
    return net_cpu


def configure(args):
    # configure CompositeField
    heads.CompositeField.dropout_p = args.head_dropout
    heads.CompositeField.quad = args.head_quad
    heads.CompositeFieldFused.dropout_p = args.head_dropout
    heads.CompositeFieldFused.quad = args.head_quad


def cli(parser):
    group = parser.add_argument_group('network configuration')
    group.add_argument('--checkpoint', default=None,
                       help=('Load a model from a checkpoint. '
                             'Use "resnet50", "shufflenetv2k16w" '
                             'or "shufflenetv2k30w" for pretrained OpenPifPaf models.'))
    group.add_argument('--num-classes', default=None, type=int)
    group.add_argument('--basenet', default=None,
                       help='base network, e.g. resnet50')
    group.add_argument('--headnets', default=None, nargs='+',
                       help='head networks')
    group.add_argument('--no-pretrain', dest='pretrained', default=True, action='store_false',
                       help='create model without ImageNet pretraining')
    group.add_argument('--two-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--multi-scale', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--no-multi-scale-hflip',
                       dest='multi_scale_hflip', default=True, action='store_false',
                       help='[experimental]')
    group.add_argument('--cross-talk', default=0.0, type=float,
                       help='[experimental]')
    group.add_argument('--no-download-progress', dest='download_progress',
                       default=True, action='store_false',
                       help='suppress model download progress bar')
    group.add_argument('--with-edge', default=False, action='store_true')

    group = parser.add_argument_group('head')
    group.add_argument('--head-dropout', default=heads.CompositeFieldFused.dropout_p, type=float,
                       help='[experimental] zeroing probability of feature in head input')
    group.add_argument('--head-quad', default=heads.CompositeFieldFused.quad, type=int,
                       help='number of times to apply quad (subpixel conv) to heads')
