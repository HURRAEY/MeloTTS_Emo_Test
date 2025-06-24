#!/bin/bash

# 감정 지원 MeloTTS 훈련 스크립트

echo "🎵 감정 지원 MeloTTS 훈련 시작..."

# 가상환경 활성화
source emotion_env/bin/activate

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/MeloTTS"

# 디렉토리 생성
mkdir -p logs
mkdir -p checkpoints

# 훈련 실행
cd MeloTTS

python melo/train_emotion.py \
    -c melo/configs/config.json \
    -m emotion_melotts \
    -e ../emotion_config.json 2>&1 | tee ../logs/training.log

echo "✅ 훈련 완료!" 