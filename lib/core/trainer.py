# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import torch
import torch.nn.functional as F

from utils.utils import AverageMeter
from utils.vis import save_debug_images


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, heatmaps, masks, joints) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(images)

        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
        joints = list(map(lambda x: x.cuda(non_blocking=True), joints))

        # loss = loss_factory(outputs, heatmaps, masks)
        heatmaps_losses, push_losses, pull_losses = \
            loss_factory(outputs, heatmaps, masks, joints)

        loss = 0
        for idx in range(cfg.LOSS.NUM_STAGES):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heatmaps_loss_meter[idx].update(
                    heatmaps_loss.item(), images.size(0)
                )
                loss = loss + heatmaps_loss
                if push_losses[idx] is not None:
                    push_loss = push_losses[idx].mean(dim=0)
                    push_loss_meter[idx].update(
                        push_loss.item(), images.size(0)
                    )
                    loss = loss + push_loss
                if pull_losses[idx] is not None:
                    pull_loss = pull_losses[idx].mean(dim=0)
                    pull_loss_meter[idx].update(
                        pull_loss.item(), images.size(0)
                    )
                    loss = loss + pull_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull')
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(i),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )

def do_train_da(cfg, model, data_loader, data_target_loader, loss_factory, optimizer, epoch,
             output_dir, tb_log_dir, writer_dict, fp16=False):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmaps_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    push_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    pull_loss_meter = [AverageMeter() for _ in range(cfg.LOSS.NUM_STAGES)]
    domain_loss_meter = [AverageMeter()]

    domain_loss_weight = cfg.LOSS.DOMAIN_LOSS_FACTOR

    # switch to train mode
    model.train()

    n_batch = min(len(data_loader), len(data_target_loader))
    iter_source, iter_target = iter(data_loader), iter(data_target_loader)

    end = time.time()
    for i in range(n_batch-1):
        images, heatmaps, masks, joints = next(iter_source)
        images_target, _, _, _ = next(iter_target)

        data_time.update(time.time() - end)

        with torch.autograd.set_detect_anomaly(True):
            b_s = images.size(0)
            imgs = torch.cat((images, images_target),dim=0)
            outputs_all, feats = model(imgs)

            # outputs, _ = outputs_all.split(b_s, dim=0)
            outputs = []
            for output in outputs_all:
                out, _ = output.split(b_s, dim=0)
                outputs.append(out)

            source_feats, target_feats = feats.split(b_s, dim=0)
            
            # source_feats.sum.backward()

            # target_outputs, target_feats = model(images_target)
            # import pdb;pdb.set_trace()

            heatmaps = list(map(lambda x: x[:, cfg.MODEL.TRAIN_CHANNEL,:,:].cuda(non_blocking=True), heatmaps))
            masks = list(map(lambda x: x.cuda(non_blocking=True), masks))
            joints = list(map(lambda x: x[:, cfg.MODEL.TRAIN_CHANNEL, :].cuda(non_blocking=True), joints))


            # pose loss
            heatmaps_losses, push_losses, pull_losses = \
                loss_factory(outputs, heatmaps, masks, joints)

            loss = 0
            for idx in range(cfg.LOSS.NUM_STAGES):
                if heatmaps_losses[idx] is not None:
                    heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                    heatmaps_loss_meter[idx].update(
                        heatmaps_loss.item(), images.size(0)
                    )
                    loss = loss + heatmaps_loss
                    if push_losses[idx] is not None:
                        push_loss = push_losses[idx].mean(dim=0)
                        push_loss_meter[idx].update(
                            push_loss.item(), images.size(0)
                        )
                        loss = loss + push_loss
                    if pull_losses[idx] is not None:
                        pull_loss = pull_losses[idx].mean(dim=0)
                        pull_loss_meter[idx].update(
                            pull_loss.item(), images.size(0)
                        )
                        loss = loss + pull_loss

            source_label = torch.ones_like(source_feats).cuda(non_blocking=True)
            domain_loss_source = F.binary_cross_entropy_with_logits(source_feats, source_label)
            target_label = torch.zeros_like(target_feats).cuda(non_blocking=True)
            domain_loss_target = F.binary_cross_entropy_with_logits(target_feats, target_label)

            domain_loss = 0.5 * domain_loss_weight[0] * (domain_loss_source + domain_loss_target)
            # domain_loss = domain_loss_weight[0] * domain_loss_source
            domain_loss_meter[0].update(domain_loss, images.size(0))
            loss = loss + domain_loss

            # compute gradient and do update step
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}{domain_loss}'.format(
                      epoch, i, n_batch,
                      batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmaps_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                      domain_loss=_get_loss_info(domain_loss_meter, 'domain'),
                  )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            for idx in range(cfg.LOSS.NUM_STAGES):
                writer.add_scalar(
                    'train_stage{}_heatmaps_loss'.format(i),
                    heatmaps_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_push_loss'.format(idx),
                    push_loss_meter[idx].val,
                    global_steps
                )
                writer.add_scalar(
                    'train_stage{}_pull_loss'.format(idx),
                    pull_loss_meter[idx].val,
                    global_steps
                )
            writer.add_scalar(
                'train_domain_loss',
                domain_loss_meter[0].val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            for scale_idx in range(len(outputs)):
                prefix_scale = prefix + '_output_{}'.format(
                    cfg.DATASET.OUTPUT_SIZE[scale_idx]
                )
                save_debug_images(
                    cfg, images, heatmaps[scale_idx], masks[scale_idx],
                    outputs[scale_idx], prefix_scale
                )

def _get_loss_info(loss_meters, loss_name):
    msg = ''
    for i, meter in enumerate(loss_meters):
        msg += 'Stage{i}-{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            i=i, name=loss_name, meter=meter
        )

    return msg
