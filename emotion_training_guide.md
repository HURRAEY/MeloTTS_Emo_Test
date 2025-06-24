# 🎯 Applio → MeloTTS 감정 파인튜닝 완전 가이드

## 📋 **준비 사항**

### 1. **Applio 데이터 구조 확인**

Applio로 전처리된 데이터가 다음 중 어떤 형태인지 확인해주세요:

#### **패턴 A: 감정별 폴더 구조**

```
applio_output/
├── happy/
│   ├── speaker1_happy_001.wav
│   ├── speaker1_happy_002.wav
│   └── ...
├── sad/
│   ├── speaker1_sad_001.wav
│   └── ...
├── angry/
│   └── ...
└── metadata.csv (선택사항)
```

#### **패턴 B: 플랫 구조 + 메타데이터**

```
applio_output/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

#### **패턴 C: 텍스트 파일 리스트**

```
applio_output/
├── wavs/
└── filelist.txt  # "파일경로|텍스트|감정" 형태
```

### 2. **필요한 정보 수집**

다음 정보를 알려주시면 맞춤형 변환 스크립트를 제공할 수 있습니다:

```bash
# 1. Applio 출력 디렉토리 확인
ls -la /path/to/your/applio/output/

# 2. 메타데이터 파일 확인 (있다면)
head -10 metadata.csv
# 또는
head -10 filelist.txt

# 3. 감정별 파일 수 확인
find /path/to/applio/output -name "*.wav" | wc -l
```

## 🔧 **단계별 변환 과정**

### **1단계: 데이터 변환**

```bash
# Applio 데이터를 MeloTTS 형식으로 변환
python applio_to_melotts_converter.py \
    --applio_dir /path/to/your/applio/output \
    --output_dir ./emotion_dataset \
    --language KR  # 또는 EN, JP 등
```

#### **변환 결과**

```
emotion_dataset/
├── wavs/
│   ├── happy_0001.wav
│   ├── happy_0002.wav
│   ├── sad_0001.wav
│   └── ...
├── metadata.list
└── config.json
```

### **2단계: 데이터셋 분할**

```bash
# 학습/검증 데이터 분할 (90:10 비율)
python split_dataset.py \
    --metadata emotion_dataset/metadata.list \
    --output_dir emotion_dataset \
    --val_ratio 0.1
```

### **3단계: MeloTTS 모델 수정**

#### **A. 감정 임베딩 추가**

`MeloTTS/melo/models.py`의 `SynthesizerTrn` 클래스 수정:

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, ..., n_emotions=7, **kwargs):
        super().__init__()
        # ... 기존 코드 ...

        # 감정 임베딩 추가
        self.n_emotions = n_emotions
        if n_emotions > 0:
            self.emb_emotions = nn.Embedding(n_emotions, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid, tone, language,
                bert, ja_bert, emotion_id=None):

        # 스피커 임베딩
        if self.n_speakers > 0:
            g_spk = self.emb_g(sid).unsqueeze(-1)
        else:
            g_spk = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        # 감정 임베딩 추가
        if emotion_id is not None and self.n_emotions > 0:
            g_emo = self.emb_emotions(emotion_id).unsqueeze(-1)
            g = g_spk + g_emo  # 스피커 + 감정 결합
        else:
            g = g_spk

        # ... 나머지 코드는 동일 ...
```

#### **B. 데이터 로더 수정**

`MeloTTS/melo/data_utils.py` 수정:

```python
def get_audio_text_speaker_pair(self, audiopath_sid_text):
    # 감정 정보 추가 파싱
    if len(audiopath_sid_text) >= 8:  # 감정 정보 포함
        audiopath, sid, language, text, phones, tone, word2ph, emotion_id = audiopath_sid_text
        emotion_id = torch.LongTensor([int(emotion_id)])
    else:  # 기존 형태
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
        emotion_id = torch.LongTensor([0])  # neutral

    # ... 기존 처리 ...

    return (phones, spec, wav, sid, tone, language, bert, ja_bert, emotion_id)
```

### **4단계: 설정 파일 업데이트**

#### **emotion_config.json 생성**

```json
{
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 52,
    "epochs": 10000,
    "learning_rate": 0.0002,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 4,
    "fp16_run": false,
    "lr_decay": 0.999875,
    "segment_size": 16384,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "skip_optimizer": true
  },
  "data": {
    "training_files": "emotion_dataset/train.list",
    "validation_files": "emotion_dataset/val.list",
    "max_wav_value": 32768.0,
    "sampling_rate": 44100,
    "filter_length": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 1,
    "n_emotions": 7,
    "cleaned_text": true,
    "spk2id": { "speaker_00": 0 },
    "emotion_map": {
      "neutral": 0,
      "happy": 1,
      "sad": 2,
      "angry": 3,
      "surprised": 4,
      "fearful": 5,
      "disgusted": 6
    }
  },
  "model": {
    "use_spk_conditioned_encoder": true,
    "use_noise_scaled_mas": true,
    "use_mel_posterior_encoder": false,
    "use_duration_discriminator": true,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "n_layers_trans_flow": 3,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5]
    ],
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 256,
    "n_emotions": 7
  }
}
```

### **5단계: 텍스트 전처리**

```bash
cd MeloTTS/melo
python preprocess_text.py \
    --metadata ../../emotion_dataset/train.list \
    --config_path emotion_config.json \
    --val-per-spk 10 \
    --max-val-total 100
```

### **6단계: 감정 모델 훈련**

```bash
# 단일 GPU
python train.py --config emotion_config.json

# 다중 GPU (권장)
torchrun --nproc_per_node=2 train.py --config emotion_config.json
```

## 🎯 **추론 및 사용**

### **감정별 음성 합성**

```python
import torch
from melo.api import TTS

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TTS(language='KR', device=device)
model.load_checkpoint('path/to/emotion_model.pth')

# 감정별 합성
emotions = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
    'surprised': 4, 'fearful': 5, 'disgusted': 6
}

text = "안녕하세요, 오늘 날씨가 정말 좋네요!"

for emotion_name, emotion_id in emotions.items():
    audio = model.tts_to_file(
        text=text,
        speaker_id=0,
        emotion_id=emotion_id,
        output_path=f"output_{emotion_name}.wav",
        speed=1.0,
        noise_scale=0.667,
        noise_scale_w=0.8
    )
    print(f"✅ {emotion_name} 음성 생성 완료")
```

## 📊 **데이터 요구사항**

### **최소 요구사항**

- 감정당 **최소 30분** (약 600 utterances)
- 총 **3.5시간** (7개 감정)

### **권장 요구사항**

- 감정당 **2시간** (약 2400 utterances)
- 총 **14시간** (7개 감정)

### **고품질 요구사항**

- 감정당 **5시간+** (6000+ utterances)
- 총 **35시간+** (7개 감정)

## 🔧 **문제 해결**

### **Q1: Applio 파일 구조가 다른 경우**

```python
# applio_to_melotts_converter.py의 _extract_emotion_from_path 함수 수정
def _extract_emotion_from_path(self, file_path):
    path_lower = file_path.lower()

    # 사용자 맞춤 패턴 추가
    emotion_patterns = {
        'neutral': ['neutral', 'normal', '중성'],
        'happy': ['happy', 'joy', '행복', '기쁨'],
        'sad': ['sad', 'sorrow', '슬픔', '우울'],
        'angry': ['angry', 'mad', '화남', '분노'],
        # ... 추가 패턴
    }

    for emotion, patterns in emotion_patterns.items():
        if any(pattern in path_lower for pattern in patterns):
            return emotion

    return 'neutral'
```

### **Q2: 메타데이터 형식이 다른 경우**

```python
# CSV 컬럼명이 다른 경우 매핑
COLUMN_MAPPING = {
    'file_name': 'file_path',
    'transcript': 'text',
    'emotion_label': 'emotion',
    'speaker_name': 'speaker_id'
}
```

### **Q3: 훈련 중 메모리 부족**

```json
// emotion_config.json에서 배치 크기 조정
{
  "train": {
    "batch_size": 2, // 4에서 2로 감소
    "segment_size": 8192 // 16384에서 8192로 감소
  }
}
```

## 📈 **성능 평가**

### **객관적 평가**

```python
# 감정 인식 정확도 측정
from emotion2vec import Emotion2Vec

evaluator = Emotion2Vec.from_pretrained("emotion2vec/emotion2vec_plus_large")

def evaluate_emotion_accuracy(generated_audios, target_emotions):
    correct = 0
    total = len(generated_audios)

    for audio, target in zip(generated_audios, target_emotions):
        predicted = evaluator.predict(audio)
        if predicted == target:
            correct += 1

    accuracy = correct / total
    print(f"감정 인식 정확도: {accuracy:.2%}")
    return accuracy
```

### **주관적 평가**

```python
# MOS (Mean Opinion Score) 평가
def collect_mos_scores():
    emotions = ['neutral', 'happy', 'sad', 'angry']

    for emotion in emotions:
        print(f"\n{emotion} 음성 품질 평가 (1-5점):")
        # 평가자들에게 음성 재생 후 점수 수집
        pass
```

이제 **Applio 데이터의 구체적인 구조**를 알려주시면, 위 스크립트를 해당 형태에 맞게 커스터마이징해드릴 수 있습니다!
