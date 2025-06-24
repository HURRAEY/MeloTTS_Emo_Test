# MeloTTS 일본어 품질 향상 가이드

영어에는 `EN_Newest` 모델이 있어 고품질 음성을 제공하지만, 일본어는 아직 최신 모델이 없습니다. 하지만 다양한 방법으로 일본어 TTS 품질을 향상시킬 수 있습니다.

## 🎯 주요 개선 방법

### 1. 파라미터 최적화

기본 설정보다 더 나은 결과를 위한 파라미터 조정:

```python
# 기본 설정
model.tts_to_file(text, speaker_id, 'output.wav')

# 품질 향상 설정
model.tts_to_file(
    text,
    speaker_id,
    'enhanced_output.wav',
    noise_scale=0.5,      # 0.6 → 0.5 (더 안정적)
    noise_scale_w=0.7,    # 0.8 → 0.7 (노이즈 감소)
    speed=0.95,           # 1.0 → 0.95 (자연스러운 속도)
    sdp_ratio=0.2         # 기본값 유지
)
```

### 2. 텍스트 전처리 개선

```python
def optimize_japanese_text(text):
    # 전각 문자를 반각으로 변환
    text = text.replace('。', '.')
    text = text.replace('、', ',')
    text = text.replace('！', '!')
    text = text.replace('？', '?')

    # 영어-일본어 경계에 공백 추가
    import re
    text = re.sub(r'([ひらがなカタカナ一-龯])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])([ひらがなカタカナ一-龯])', r'\1 \2', text)

    return text
```

### 3. 다양한 음성 스타일

용도에 따른 파라미터 조합:

```python
# 부드러운 음성 (감정적 표현)
soft_params = {
    'noise_scale': 0.4,
    'noise_scale_w': 0.6,
    'speed': 0.9
}

# 명확한 음성 (발음 명확성)
clear_params = {
    'noise_scale': 0.6,
    'noise_scale_w': 0.8,
    'speed': 1.0,
    'sdp_ratio': 0.1
}
```

## 🚀 빠른 시작

### 설치

```bash
# MeloTTS 설치
cd MeloTTS
pip install -e .
python -m unidic download
```

### 기본 사용법

```python
from melo.api import TTS

# 모델 초기화
model = TTS(language='JP', device='auto')
speaker_id = model.hps.data.spk2id['JP']

# 기본 음성 생성
text = "こんにちは、今日はとても良い天気ですね。"
model.tts_to_file(text, speaker_id, 'output.wav')
```

### 품질 향상 사용법

```python
# 향상된 설정으로 음성 생성
enhanced_text = optimize_japanese_text(text)
model.tts_to_file(
    enhanced_text,
    speaker_id,
    'enhanced_output.wav',
    noise_scale=0.5,
    noise_scale_w=0.7,
    speed=0.95
)
```

## 📊 품질 비교

| 설정     | noise_scale | noise_scale_w | speed | 특징               |
| -------- | ----------- | ------------- | ----- | ------------------ |
| 기본     | 0.6         | 0.8           | 1.0   | 표준 품질          |
| 향상     | 0.5         | 0.7           | 0.95  | 안정적, 자연스러움 |
| 부드러움 | 0.4         | 0.6           | 0.9   | 감정적 표현        |
| 명확함   | 0.6         | 0.8           | 1.0   | 발음 명확성        |

## 🔧 고급 개선 방법

### 1. 대안 모델 고려

- **MB-iSTFT-VITS-44100-Ja**: 44100Hz 일본어 특화 모델
- **StyleBert-VITS2**: 감정 표현이 향상된 모델
- **VOICEVOX**: 일본어 전용 오픈소스 TTS

### 2. 파인튜닝

고품질 일본어 데이터셋으로 모델 개선:

- JSUT 데이터셋
- ITA コーパス
- 사용자 정의 데이터셋

### 3. 상업용 솔루션

더 높은 품질이 필요한 경우:

- Azure Cognitive Services
- Google Cloud Text-to-Speech
- Amazon Polly

## 📁 파일 구조

```
MeloTTS_Emo_Test/
├── japanese_quality_enhancement.py  # 품질 향상 클래스
├── example_usage.py                 # 사용 예시
├── README.md                        # 이 파일
└── MeloTTS/                         # MeloTTS 저장소
```

## 🎵 예시 실행

### 전체 테스트 실행

```bash
python japanese_quality_enhancement.py
```

### 간단한 비교 테스트

```bash
python example_usage.py
```

## 💡 품질 향상 팁

### 텍스트 작성 팁

1. **정확한 일본어 문법** 사용
2. **적절한 문장부호** 활용 (。、！？)
3. **긴 문장보다 짧은 문장**으로 분할
4. **외래어는 카타카나**로 정확히 표기
5. **한자 읽기가 애매한 경우** 히라가나 병기

### 파라미터 튜닝 가이드

- `noise_scale`: **0.4-0.6** (낮을수록 안정적)
- `noise_scale_w`: **0.6-0.8** (노이즈 제어)
- `speed`: **0.9-1.0** (일본어는 약간 느리게)
- `sdp_ratio`: **0.1-0.3** (낮을수록 명확)

### 발음 개선

- 숫자는 문맥에 맞게 표현 (123 → "百二十三" 또는 "いちにさん")
- 영어 단어 주변에 공백 추가
- 감탄사나 간투사 적절히 활용

## ❓ 자주 묻는 질문

**Q: 영어의 EN_Newest 같은 일본어 최신 모델이 있나요?**  
A: 현재는 없습니다. 하지만 파라미터 튜닝으로 충분히 좋은 품질을 얻을 수 있습니다.

**Q: 어떤 파라미터가 가장 중요한가요?**  
A: `noise_scale`과 `speed`가 가장 큰 영향을 미칩니다. 0.5와 0.95를 권장합니다.

**Q: 다른 일본어 TTS와 비교하면 어떤가요?**  
A: VOICEVOX나 상업용 솔루션이 더 좋을 수 있지만, MeloTTS도 적절한 설정으로 충분히 실용적입니다.

## 🤝 기여

품질 향상 방법을 발견하시면 이슈나 PR로 공유해주세요!

## 📄 라이선스

MIT License - MeloTTS 프로젝트와 동일
