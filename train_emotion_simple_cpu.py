#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# MeloTTS 경로 추가
sys.path.append('MeloTTS/melo')

from melo import commons, utils
from melo.data_utils_emotion import TextAudioSpeakerEmotionLoader, TextAudioSpeakerEmotionCollate
from melo.models_emotion import SynthesizerTrnEmotion
from melo.models import MultiPeriodDiscriminator
from melo.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

def main():
    # 설정 로드
    with open('MeloTTS/melo/configs/config.json') as f:
        config = json.load(f)
    hps = utils.HParams(**config)
    
    with open('emotion_config.json') as f:
        emotion_config = json.load(f)
    
    # 디렉토리 설정
    hps.model_dir = "checkpoints/emotion_melotts"
    os.makedirs(hps.model_dir, exist_ok=True)
    
    # CPU 디바이스 설정
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 데이터 로더 (절대 경로 사용)
    train_files = hps.data.training_files.replace('../', '')
    val_files = hps.data.validation_files.replace('../', '')
    train_dataset = TextAudioSpeakerEmotionLoader(train_files, hps.data)
    val_dataset = TextAudioSpeakerEmotionLoader(val_files, hps.data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # 작은 배치 사이즈로 시작
        shuffle=True, 
        num_workers=1,  # CPU에서는 1로 설정
        collate_fn=TextAudioSpeakerEmotionCollate(),
        pin_memory=False  # CPU에서는 False
    )
    
    # 모델 초기화
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
    
    # 최적화기
    optimizer_g = torch.optim.AdamW(net_g.parameters(), 0.0001)  # 낮은 학습률
    optimizer_d = torch.optim.AdamW(net_d.parameters(), 0.0001)
    
    # 로거
    logger = utils.get_logger(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    
    print("🎵 감정 지원 MeloTTS 훈련 시작!")
    
    # 간단한 훈련 루프
    net_g.train()
    net_d.train()
    
    for epoch in range(2):  # 2 에포크만 테스트
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 3 배치만 테스트
                break
                
            x, x_lengths, y, y_lengths, sid, emotion_ids, tone, language, bert, ja_bert = batch
            
            # CPU로 이동
            x, x_lengths = x.to(device), x_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device) 
            sid = sid.to(device)
            emotion_ids = emotion_ids.to(device)
            tone = tone.to(device)
            language = language.to(device)
            bert = bert.to(device)
            ja_bert = ja_bert.to(device)
            
            try:
                # Forward pass
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, y, y_lengths, sid, emotion_ids, tone, language, bert, ja_bert
                )
                
                # Loss 계산 (간소화)
                mel = spec_to_mel_torch(
                    y, hps.data.filter_length, hps.data.n_mel_channels, 
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
                )
                
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax
                )
                
                loss_mel = F.l1_loss(y_mel, y_hat_mel)
                loss_dur = torch.sum(l_length.float())
                loss = loss_mel + loss_dur * 0.1
                
                # Backward
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()
                
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"배치 {batch_idx}에서 오류 발생: {e}")
                continue
    
    print("✅ 테스트 훈련 완료!")
    
    # 체크포인트 저장
    checkpoint_path = os.path.join(hps.model_dir, "test_model.pth")
    torch.save({
        'model': net_g.state_dict(),
        'epoch': 2
    }, checkpoint_path)
    print(f"모델 저장: {checkpoint_path}")

if __name__ == "__main__":
    main() 