# 🎵 MeloTTS 감정 지원 완전 가이드

## 📋 개요

이 가이드는 MeloTTS에 감정 지원 기능을 추가하는 완전한 과정을 다룹니다. TDM_LLJ 데이터셋을 사용하여 기쁨, 슬픔, 화남, 우울 등의 감정을 표현할 수 있는 TTS 모델을 구축합니다.

## 🎯 완성된 결과물

### ✅ 데이터 변환 완료

- **총 WAV 파일**: 62개 (3-15초 세그먼트)
- **학습 데이터**: 50개
- **검증 데이터**: 10개
- **감정별 분포**:
  - Happy (기쁨): 16개
  - Sad (슬픔): 37개
  - Angry (화남): 9개

### ✅ 모델 수정 완료

- `SynthesizerTrnEmotion`: 감정 임베딩 지원 모델
- `EmotionEmbedding`: 감정 임베딩 레이어
- `TextAudioSpeakerEmotionLoader`: 감정 지원 데이터 로더

## 📁 프로젝트 구조

```
MeloTTS_Emo_Test/
├── emotion_dataset_tdm/          # 변환된 데이터셋
│   ├── wavs/                     # 62개 오디오 세그먼트
│   ├── metadata.list             # 전체 메타데이터
│   ├── train.list                # 학습용 데이터
│   ├── val.list                  # 검증용 데이터
│   ├── config.json               # MeloTTS 설정
│   └── conversion_log.json       # 변환 로그
├── MeloTTS/                      # MeloTTS 소스코드
│   ├── melo/
│   │   ├── models_emotion.py     # 감정 지원 모델
│   │   ├── train_emotion.py      # 감정 훈련 스크립트
│   │   └── data_utils_emotion.py # 감정 데이터 로더
├── emotion_config.json           # 감정 훈련 설정
├── train_emotion_melotts.sh      # 훈련 실행 스크립트
└── emotion_env/                  # Python 가상환경
```

## 🔧 기술적 구현

### 1. 감정 임베딩 아키텍처

```python
class EmotionEmbedding(nn.Module):
    def __init__(self, n_emotions, emotion_channels=256):
        self.emb_e = nn.Embedding(n_emotions, emotion_channels)

    def forward(self, emotion_ids):
        return self.emb_e(emotion_ids)
```

### 2. 통합된 gin_channels

- **기존**: speaker embedding (256차원)
- **새로운**: speaker + emotion embedding (512차원)
- **결합 방식**: `torch.cat([speaker_g, emotion_g], dim=1)`

### 3. 감정 매핑

```python
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,      # 기쁨
    "sad": 2,        # 슬픔
    "angry": 3,      # 화남
    "depressed": 4,  # 우울
    "surprised": 5,  # 놀람
    "fearful": 6     # 두려움
}
```

## 🚀 훈련 실행 방법

### 1. 환경 설정

```bash
# 가상환경 활성화
source emotion_env/bin/activate

# 필요한 라이브러리 설치 (이미 완료됨)
pip install torch torchaudio librosa soundfile pydub
```

### 2. 훈련 실행

```bash
# 훈련 스크립트 실행
./train_emotion_melotts.sh

# 또는 직접 실행
cd MeloTTS
python melo/train_emotion.py \
    -c configs/base.json \
    -m emotion_melotts \
    -e ../emotion_config.json \
    --pretrained ../pretrained_model.pth
```

### 3. 훈련 모니터링

```bash
# TensorBoard로 훈련 과정 모니터링
tensorboard --logdir logs/

# 로그 확인
tail -f logs/training.log
```

## 📊 데이터 품질 분석

### 세그먼트 분할 결과

- **원본 파일**: 30분 연속 녹음 (329MB)
- **분할 결과**: 3-15초 세그먼트 (약 600KB)
- **분할 기준**: 무음 기반 (-40dB, 1초 무음)
- **품질 필터**: 너무 짧거나 긴 세그먼트 제거

### 감정별 데이터 품질

- **Happy**: 16개 세그먼트 (기쁨 표현 명확)
- **Sad**: 37개 세그먼트 (슬픔 표현 풍부)
- **Angry**: 9개 세그먼트 (화남 표현 강렬)

## 🎯 예상 성능

### 훈련 전략

1. **기존 모델 파인튜닝**: 사전 훈련된 MeloTTS 모델 로드
2. **감정 임베딩 초기화**: 랜덤 초기화 후 훈련
3. **점진적 학습**: 낮은 학습률로 안정적 훈련

### 예상 결과

- **감정 인식 정확도**: 85%+
- **음질 유지**: 기존 MeloTTS 수준
- **감정 표현력**: 명확한 감정 차별화

## 🔍 모니터링 지표

### 훈련 중 확인할 지표

1. **Loss 감소**: generator loss, discriminator loss
2. **감정별 성능**: 각 감정별 재생 품질
3. **음질 지표**: mel-spectrogram 유사도
4. **수렴 상태**: validation loss 안정화

### 평가 방법

```python
# 감정별 테스트
emotions = ["happy", "sad", "angry", "depressed"]
for emotion in emotions:
    emotion_id = EMOTION_MAP[emotion]
    audio = model.infer(text, speaker_id, emotion_id)
    # 품질 평가 및 저장
```

## 🛠️ 문제 해결

### 일반적인 문제들

1. **메모리 부족**: batch_size 줄이기
2. **수렴 안됨**: learning_rate 조정
3. **감정 구분 안됨**: 데이터 품질 확인
4. **음질 저하**: 훈련 데이터 검증

### 디버깅 팁

```bash
# GPU 메모리 확인
nvidia-smi

# 훈련 로그 분석
grep "Loss" logs/training.log

# 모델 크기 확인
python -c "import torch; model = torch.load('checkpoint.pth'); print(model.keys())"
```

## 📈 향후 개선 방향

### 1. 데이터 확장

- 더 많은 감정 데이터 수집
- 감정 강도 레이블링 추가
- 다국어 감정 데이터

### 2. 모델 개선

- 감정 벡터 산술 연산
- DPO for Emotions 적용
- Multi-modal Prompt 활용

### 3. 평가 체계

- 감정 인식 정확도 측정
- 주관적 품질 평가
- A/B 테스트

## 🎉 완료된 작업

### ✅ 데이터 전처리

- [x] TDM_LLJ 데이터 분석
- [x] 무음 기반 세그먼트 분할
- [x] 감정별 라벨링
- [x] MeloTTS 형식 변환

### ✅ 모델 개발

- [x] 감정 임베딩 레이어 구현
- [x] SynthesizerTrnEmotion 모델
- [x] 감정 지원 데이터 로더
- [x] 훈련 스크립트 작성

### ✅ 설정 및 스크립트

- [x] 감정 설정 파일
- [x] 훈련 실행 스크립트
- [x] 환경 설정 완료

## 🚀 다음 단계

1. **사전 훈련 모델 준비**: MeloTTS 기반 모델 다운로드
2. **훈련 실행**: `./train_emotion_melotts.sh` 실행
3. **성능 평가**: 감정별 테스트 및 품질 평가
4. **모델 배포**: 최종 모델 저장 및 배포

---

**🎵 감정이 담긴 TTS의 세계로 여러분을 초대합니다!**
