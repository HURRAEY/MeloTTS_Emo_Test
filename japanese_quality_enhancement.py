"""
MeloTTS 일본어 품질 향상 가이드

영어에는 EN_Newest가 있지만, 일본어의 품질을 향상시키기 위한 방법들:

1. 파라미터 조정 최적화
2. 텍스트 전처리 개선
3. 음성 합성 설정 최적화
4. BERT 모델 활용 최적화
"""

from melo.api import TTS
import torch
import numpy as np
import re

class JapaneseQualityEnhancer:
    def __init__(self, device='auto'):
        """일본어 TTS 품질 향상을 위한 클래스"""
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): 
                device = 'cuda'
            if torch.backends.mps.is_available(): 
                device = 'mps'
        
        self.device = device
        self.model = TTS(language='JP', device=device)
        self.speaker_id = self.model.hps.data.spk2id['JP']
        
    def optimize_text_preprocessing(self, text):
        """일본어 텍스트 전처리 최적화"""
        # 1. 전각 문자를 반각으로 변환
        text = text.replace('。', '.')
        text = text.replace('、', ',')
        text = text.replace('！', '!')
        text = text.replace('？', '?')
        
        # 2. 숫자를 일본어로 변환 (예: 123 -> 百二十三)
        # 이는 MeCab에서 자동으로 처리되지만, 명시적으로 처리할 수도 있음
        
        # 3. 영어 단어 주변에 공백 추가 (자연스러운 발음을 위해)
        text = re.sub(r'([ひらがなカタカナ一-龯])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])([ひらがなカタカナ一-龯])', r'\1 \2', text)
        
        return text
    
    def generate_with_enhanced_quality(self, text, output_path=None, **kwargs):
        """품질 향상된 일본어 음성 생성"""
        # 텍스트 전처리
        enhanced_text = self.optimize_text_preprocessing(text)
        
        # 최적화된 파라미터 설정
        enhanced_params = {
            'sdp_ratio': 0.2,      # SDP 비율 (기본값)
            'noise_scale': 0.5,    # 노이즈 스케일 (0.6에서 0.5로 감소 - 더 안정적)
            'noise_scale_w': 0.7,  # 노이즈 스케일 w (0.8에서 0.7로 감소)
            'speed': 0.95,         # 속도 (1.0에서 0.95로 약간 감소 - 더 자연스러운 발음)
            'quiet': False
        }
        
        # 사용자 정의 파라미터가 있으면 덮어쓰기
        enhanced_params.update(kwargs)
        
        return self.model.tts_to_file(
            enhanced_text, 
            self.speaker_id, 
            output_path=output_path,
            **enhanced_params
        )
    
    def generate_high_quality_comparison(self, text):
        """기본 설정과 향상된 설정의 비교"""
        print("=== 일본어 TTS 품질 비교 ===")
        
        # 기본 설정
        print("1. 기본 설정으로 생성 중...")
        self.model.tts_to_file(text, self.speaker_id, 'japanese_default.wav')
        
        # 향상된 설정
        print("2. 품질 향상 설정으로 생성 중...")
        self.generate_with_enhanced_quality(text, 'japanese_enhanced.wav')
        
        # 다양한 파라미터 조합 테스트
        print("3. 다양한 설정 테스트 중...")
        
        # 더 부드러운 음성 (감정적 표현 강화)
        self.generate_with_enhanced_quality(
            text, 
            'japanese_soft.wav',
            noise_scale=0.4,
            noise_scale_w=0.6,
            speed=0.9
        )
        
        # 더 명확한 음성 (발음 명확성 강화)
        self.generate_with_enhanced_quality(
            text, 
            'japanese_clear.wav',
            noise_scale=0.6,
            noise_scale_w=0.8,
            speed=1.0,
            sdp_ratio=0.1
        )
        
        print("음성 파일들이 생성되었습니다:")
        print("- japanese_default.wav: 기본 설정")
        print("- japanese_enhanced.wav: 품질 향상 설정")
        print("- japanese_soft.wav: 부드러운 음성")
        print("- japanese_clear.wav: 명확한 음성")

def main():
    """메인 실행 함수"""
    # 일본어 품질 향상 클래스 초기화
    enhancer = JapaneseQualityEnhancer()
    
    # 테스트 텍스트
    test_texts = [
        "こんにちは、今日はとても良い天気ですね。",
        "人工知能技術の発展により、音声合成の品質が大幅に向上しました。",
        "MeloTTSを使用して、自然な日本語音声を生成しています。",
        "123個のリンゴがあります。とても美味しそうです。"
    ]
    
    print("=== MeloTTS 일본어 품질 향상 가이드 ===\n")
    
    print("주요 개선 방법:")
    print("1. 파라미터 최적화:")
    print("   - noise_scale: 0.6 → 0.5 (더 안정적인 음성)")
    print("   - noise_scale_w: 0.8 → 0.7 (노이즈 감소)")
    print("   - speed: 1.0 → 0.95 (더 자연스러운 속도)")
    print()
    
    print("2. 텍스트 전처리:")
    print("   - 문장부호 정규화")
    print("   - 영어-일본어 경계 공백 추가")
    print("   - 숫자 표현 최적화")
    print()
    
    print("3. BERT 모델 활용:")
    print("   - tohoku-nlp/bert-base-japanese-v3 사용")
    print("   - 문맥 정보를 활용한 자연스러운 발음")
    print()
    
    # 각 텍스트에 대해 품질 향상 테스트
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== 테스트 {i}: {text} ===")
        enhancer.generate_high_quality_comparison(text)
    
    print("\n=== 추가 개선 방법 ===")
    print("1. EN_Newest처럼 일본어에도 더 나은 모델이 있는지 확인:")
    print("   - 현재는 'JP' 모델만 공식적으로 제공됨")
    print("   - 커뮤니티에서 제작한 향상된 일본어 모델 확인")
    print()
    
    print("2. 파인튜닝:")
    print("   - 고품질 일본어 데이터셋으로 모델 파인튜닝")
    print("   - JSUT, ITAコーパス 등 활용")
    print()
    
    print("3. 대안 고려:")
    print("   - MB-iSTFT-VITS-44100-Ja (44100Hz 일본어 특화)")
    print("   - StyleBert-VITS2 (감정 표현 향상)")
    print("   - VOICEVOX (일본어 특화 오픈소스)")

# 개별 기능 테스트를 위한 함수들
def test_enhanced_parameters():
    """향상된 파라미터만 테스트"""
    enhancer = JapaneseQualityEnhancer()
    text = "こんにちは、音声合成のテストです。"
    
    print("파라미터별 음질 테스트...")
    
    # SDP 비율 테스트
    for sdp in [0.1, 0.2, 0.3]:
        enhancer.generate_with_enhanced_quality(
            text, f'jp_sdp_{sdp}.wav', sdp_ratio=sdp
        )
    
    # 노이즈 스케일 테스트
    for noise in [0.3, 0.5, 0.7]:
        enhancer.generate_with_enhanced_quality(
            text, f'jp_noise_{noise}.wav', noise_scale=noise
        )
    
    # 속도 테스트
    for speed in [0.8, 0.95, 1.1]:
        enhancer.generate_with_enhanced_quality(
            text, f'jp_speed_{speed}.wav', speed=speed
        )

def get_quality_tips():
    """일본어 TTS 품질 향상 팁"""
    return """
    === MeloTTS 일본어 품질 향상 팁 ===
    
    1. 파라미터 최적화:
       - noise_scale: 0.4-0.6 (낮을수록 안정적)
       - noise_scale_w: 0.6-0.8 (노이즈 제어)
       - speed: 0.9-1.0 (일본어는 약간 느리게)
       - sdp_ratio: 0.1-0.3 (낮을수록 명확)
    
    2. 텍스트 품질:
       - 정확한 일본어 문법 사용
       - 적절한 문장부호 활용
       - 긴 문장보다 짧은 문장으로 분할
    
    3. 발음 개선:
       - 외래어는 카타카나로 정확히 표기
       - 한자 읽기가 애매한 경우 히라가나 병기
       - 숫자는 문맥에 맞게 표현
    
    4. 대안 모델 고려:
       - MeloTTS 이외의 일본어 특화 모델
       - VOICEVOX, CoeFont 등 일본어 전용 TTS
       - 상업용 솔루션: Azure Cognitive Services, Google Cloud TTS
    """

if __name__ == "__main__":
    main() 