"""
MeloTTS 영어 (EN_Newest) 및 일본어 품질 향상 사용 예시
"""

from melo.api import TTS
import torch

def setup_models():
    """영어와 일본어 모델 설정"""
    device = 'auto'
    if device == 'auto':
        device = 'cpu'
        if torch.cuda.is_available(): 
            device = 'cuda'
        if torch.backends.mps.is_available(): 
            device = 'mps'
    
    # 영어 최신 모델 (EN_Newest)
    en_model = TTS(language='EN_NEWEST', device=device)
    en_speaker_id = en_model.hps.data.spk2id['EN-Newest']
    
    # 일본어 모델
    jp_model = TTS(language='JP', device=device)
    jp_speaker_id = jp_model.hps.data.spk2id['JP']
    
    return en_model, en_speaker_id, jp_model, jp_speaker_id

def generate_english_speech():
    """영어 음성 생성 (EN_Newest 사용)"""
    en_model, en_speaker_id, _, _ = setup_models()
    
    english_text = "Hello! This is a test of the newest English model in MeloTTS."
    
    # EN_Newest 모델로 영어 음성 생성
    en_model.tts_to_file(
        english_text, 
        en_speaker_id, 
        'english_newest.wav',
        speed=1.0,
        noise_scale=0.6,
        noise_scale_w=0.8
    )
    
    print("영어 음성이 'english_newest.wav'에 저장되었습니다.")

def generate_enhanced_japanese_speech():
    """향상된 일본어 음성 생성"""
    _, _, jp_model, jp_speaker_id = setup_models()
    
    japanese_text = "こんにちは。MeloTTSを使用して高品質な日本語音声を生成しています。"
    
    # 기본 설정
    jp_model.tts_to_file(
        japanese_text, 
        jp_speaker_id, 
        'japanese_default.wav'
    )
    
    # 품질 향상 설정
    jp_model.tts_to_file(
        japanese_text, 
        jp_speaker_id, 
        'japanese_enhanced.wav',
        speed=0.95,           # 약간 느리게
        noise_scale=0.5,      # 노이즈 감소
        noise_scale_w=0.7,    # 더 안정적
        sdp_ratio=0.2
    )
    
    print("일본어 음성이 생성되었습니다:")
    print("- japanese_default.wav: 기본 설정")
    print("- japanese_enhanced.wav: 품질 향상 설정")

def compare_models():
    """영어와 일본어 모델 비교"""
    print("=== MeloTTS 모델 비교 ===\n")
    
    print("1. 영어 (EN_Newest):")
    print("   - 최신 모델로 높은 품질")
    print("   - 다양한 액센트 지원 (US, UK, AU, IN)")
    print("   - 자연스러운 억양과 발음")
    
    print("\n2. 일본어 (JP):")
    print("   - 기본 JP 모델 사용")
    print("   - EN_Newest 같은 최신 버전은 아직 없음")
    print("   - 파라미터 튜닝으로 품질 향상 가능")
    
    print("\n3. 일본어 품질 향상 방법:")
    print("   - noise_scale: 0.6 → 0.5")
    print("   - noise_scale_w: 0.8 → 0.7") 
    print("   - speed: 1.0 → 0.95")
    print("   - 텍스트 전처리 최적화")

def main():
    """메인 실행 함수"""
    print("=== MeloTTS 영어 & 일본어 품질 비교 ===\n")
    
    # 모델 비교 정보 출력
    compare_models()
    
    print("\n" + "="*50)
    print("음성 생성 시작...\n")
    
    # 영어 음성 생성
    print("1. 영어 음성 생성 (EN_Newest)")
    generate_english_speech()
    
    print("\n2. 일본어 음성 생성 (품질 향상)")
    generate_enhanced_japanese_speech()
    
    print("\n" + "="*50)
    print("완료! 생성된 파일들을 비교해보세요.")
    print("\n추천 설정:")
    print("- 영어: EN_Newest 모델 사용 (기본 설정도 충분히 좋음)")
    print("- 일본어: 파라미터 튜닝으로 품질 향상")
    print("  * noise_scale=0.5, speed=0.95 추천")

if __name__ == "__main__":
    main() 