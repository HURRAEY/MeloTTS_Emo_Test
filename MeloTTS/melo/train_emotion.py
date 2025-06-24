#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
감정 지원 MeloTTS 훈련 스크립트
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

# MeloTTS imports
from melo import commons
from melo import utils
from melo.data_utils_emotion import (
    TextAudioSpeakerEmotionLoader,
    TextAudioSpeakerEmotionCollate,
    DistributedBucketSampler
)
from melo.models_emotion import SynthesizerTrnEmotion
from melo.models import MultiPeriodDiscriminator
from melo.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from melo.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# 감정 매핑
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "depressed": 4,
    "surprised": 5,
    "fearful": 6
}

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """체크포인트 로드"""
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print(f"Warning: {k} not found in checkpoint")
            new_state_dict[k] = v
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    
    if optimizer is not None and 'optimizer' in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    return checkpoint_dict.get('iteration', 0)


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """체크포인트 저장"""
    print(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        'model': state_dict,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
    }, checkpoint_path)


def train_emotion_model(
    hps,
    train_dataset,
    val_dataset,
    model,
    net_d,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
    train_loader,
    val_loader,
    logger,
    writer,
    checkpoint_path,
    emotion_config
):
    """감정 지원 모델 훈련"""
    
    global_step = 0
    
    for epoch in range(hps.train.epochs):
        model.train()
        net_d.train()
        
        for batch_idx, (x, x_lengths, y, y_lengths, wav, wav_lengths, sid, emotion_ids, tone, language, bert, ja_bert) in enumerate(train_loader):
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)
            sid = sid.cuda(non_blocking=True)
            emotion_ids = emotion_ids.cuda(non_blocking=True)  # 감정 ID 추가
            tone = tone.cuda(non_blocking=True)
            language = language.cuda(non_blocking=True)
            bert = bert.cuda(non_blocking=True)
            ja_bert = ja_bert.cuda(non_blocking=True)
            
            # Generator forward
            with torch.cuda.amp.autocast(enabled=hps.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = model(
                    x, x_lengths, y, y_lengths, sid, emotion_ids, tone, language, bert, ja_bert
                )
                
                mel = spec_to_mel_torch(
                    y, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), 
                                                hps.data.filter_length,
                                                hps.data.n_mel_channels, 
                                                hps.data.sampling_rate,
                                                hps.data.hop_length, 
                                                hps.data.win_length,
                                                hps.data.mel_fmin, 
                                                hps.data.mel_fmax
                                               )
                
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 
                
                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with torch.no_grad():
                    loss_dr, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_dr_f, _ = feature_loss(y_d_hat_r, y_d_hat_g)
                
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = loss_fm * hps.train.c_fm
                loss_gen = loss_gen * hps.train.c_gen
                loss_dur = torch.sum(l_length.float())
                loss = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hps.train.grad_clip_thresh)
            scaler.step(optimizer_g)
            scaler.update()
            
            # Logging
            if global_step % hps.train.log_interval == 0:
                lr = optimizer_g.param_groups[0]['lr']
                losses = [loss_dr, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(f'Train Epoch: {epoch} [{batch_idx * hps.train.batch_size}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} '
                          f'LR: {lr:.6f}')
                
                scalar_dict = {"loss/g/total": loss_gen, "loss/d/total": loss_dr, "learning_rate": lr}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
                
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(loss_dr_f)})
                image_dict = { 
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    }
                utils.summarize(
                  writer=writer,
                  global_step=global_step, 
                  images=image_dict,
                  scalars=scalar_dict)
            
            global_step += 1
            
            # Validation
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, model, val_loader, writer, global_step, emotion_config)
            
            # Save checkpoint
            if global_step % hps.train.save_interval == 0:
                save_checkpoint(model, optimizer_g, scheduler_g.get_last_lr()[0], global_step, checkpoint_path)
        
        scheduler_g.step()
        scheduler_d.step()


def evaluate(hps, model, val_loader, writer, global_step, emotion_config):
    """검증"""
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, y, y_lengths, wav, wav_lengths, sid, emotion_ids, tone, language, bert, ja_bert) in enumerate(val_loader):
            if batch_idx >= 1:  # 첫 번째 배치만 평가
                break
                
            x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
            y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)
            sid = sid.cuda(non_blocking=True)
            emotion_ids = emotion_ids.cuda(non_blocking=True)
            tone = tone.cuda(non_blocking=True)
            language = language.cuda(non_blocking=True)
            bert = bert.cuda(non_blocking=True)
            ja_bert = ja_bert.cuda(non_blocking=True)
            
            # Inference
            y_hat, attn, mask, *_ = model.infer(
                x, x_lengths, sid, emotion_ids, tone, language, bert, ja_bert
            )
            
            # Mel spectrogram
            mel = spec_to_mel_torch(
                y, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), 
                                            hps.data.filter_length,
                                            hps.data.n_mel_channels, 
                                            hps.data.sampling_rate,
                                            hps.data.hop_length, 
                                            hps.data.win_length,
                                            hps.data.mel_fmin, 
                                            hps.data.mel_fmax
                                           )
            
            # Logging
            image_dict = {
                "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            }
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars={}
            )
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-e', '--emotion_config', type=str, required=True,
                        help='Emotion configuration file')
    parser.add_argument('-p', '--pretrained', type=str, default=None,
                        help='Pretrained model path')
    
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    hps = utils.HParams(**json_config)
    
    with open(args.emotion_config) as f:
        emotion_config = json.load(f)
    
    # Set model directory
    hps.model_dir = f"../checkpoints/{args.model}"
    
    # Create directories
    os.makedirs(hps.model_dir, exist_ok=True)
    
    # Set device
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)
    torch.cuda.manual_seed_all(hps.train.seed)
    
    # Data loaders
    train_dataset = TextAudioSpeakerEmotionLoader(hps.data.training_files, hps.data)
    val_dataset = TextAudioSpeakerEmotionLoader(hps.data.validation_files, hps.data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hps.train.batch_size,
        shuffle=False, 
        num_workers=8,
        collate_fn=TextAudioSpeakerEmotionCollate(),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=TextAudioSpeakerEmotionCollate(),
        pin_memory=True
    )
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    net_g = SynthesizerTrnEmotion(
        len(hps.data.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_emotions=emotion_config['model']['n_emotions'],
        emotion_channels=emotion_config['model']['emotion_channels'],
        **hps.model
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    
    # Load pretrained model if specified
    if args.pretrained:
        load_checkpoint(args.pretrained, net_g)
    
    # Optimizers
    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    optimizer_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    
    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=hps.train.lr_decay)
    
    # Scaler for mixed precision (only if CUDA is available)
    scaler = torch.cuda.amp.GradScaler(enabled=hps.train.fp16_run and torch.cuda.is_available())
    
    # Logger and writer
    logger = utils.get_logger(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    
    # Training
    checkpoint_path = os.path.join(hps.model_dir, f"{args.model}.pth")
    train_emotion_model(
        hps, train_dataset, val_dataset, net_g, net_d, optimizer_g, optimizer_d,
        scheduler_g, scheduler_d, scaler, train_loader, val_loader,
        logger, writer, checkpoint_path, emotion_config
    )


if __name__ == "__main__":
    main() 