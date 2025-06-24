#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeloTTS 감정 데이터셋을 학습/검증 세트로 분할하는 스크립트
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

def split_emotion_dataset(metadata_file, output_dir, val_ratio=0.1, seed=42):
    """
    감정별로 균등하게 학습/검증 데이터를 분할
    
    Args:
        metadata_file: metadata.list 파일 경로
        output_dir: 출력 디렉토리
        val_ratio: 검증 데이터 비율 (기본: 10%)
        seed: 랜덤 시드
    """
    
    random.seed(seed)
    
    # 메타데이터 로드
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 감정별로 데이터 그룹화
    emotion_groups = defaultdict(list)
    
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 5:
            emotion_id = parts[4]
            emotion_groups[emotion_id].append(line)
    
    train_lines = []
    val_lines = []
    
    print("📊 감정별 데이터 분할:")
    
    for emotion_id, emotion_lines in emotion_groups.items():
        # 섞기
        random.shuffle(emotion_lines)
        
        # 분할점 계산
        val_count = max(1, int(len(emotion_lines) * val_ratio))
        train_count = len(emotion_lines) - val_count
        
        # 분할
        emotion_val = emotion_lines[:val_count]
        emotion_train = emotion_lines[val_count:]
        
        train_lines.extend(emotion_train)
        val_lines.extend(emotion_val)
        
        print(f"   감정 {emotion_id}: 학습 {train_count}개, 검증 {val_count}개")
    
    # 다시 섞기
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    
    # 저장
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train.list"
    val_file = output_dir / "val.list"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    print(f"\n✅ 분할 완료!")
    print(f"📁 학습 데이터: {train_file} ({len(train_lines)}개)")
    print(f"📁 검증 데이터: {val_file} ({len(val_lines)}개)")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description="감정 데이터셋 분할")
    parser.add_argument("--metadata", required=True, help="metadata.list 파일 경로")
    parser.add_argument("--output_dir", default=".", help="출력 디렉토리")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="검증 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    args = parser.parse_args()
    
    split_emotion_dataset(
        metadata_file=args.metadata,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 