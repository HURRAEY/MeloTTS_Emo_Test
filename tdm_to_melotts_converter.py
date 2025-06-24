#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDM_LLJ (Applio RVC 전처리) 데이터를 MeloTTS 형식으로 변환하는 스크립트
"""

import os
import json
import shutil
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import argparse

# 감정 매핑 (한국어 → 영어 → ID)
EMOTION_MAP = {
    # 한국어
    '기쁨': 'happy',
    '슬픔': 'sad', 
    '화남': 'angry',
    '우울': 'depressed',
    '중성': 'neutral',
    # 영어 (이미 있는 경우)
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'joy': 'happy',
    'depression': 'depressed',
    'neutral': 'neutral'
}

EMOTION_ID_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'depressed': 4,
    'surprised': 5,
    'fearful': 6
}

class TDMToMeloTTSConverter:
    def __init__(self, tdm_dir, output_dir, language='KR', segment_length=10):
        """
        Args:
            tdm_dir: TDM_LLJ 디렉토리 경로
            output_dir: 출력 디렉토리
            language: 언어 코드 (KR, EN, JP)
            segment_length: 세그먼트 길이 (초)
        """
        self.tdm_dir = Path(tdm_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        self.segment_length = segment_length
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "wavs").mkdir(exist_ok=True)
        
    def convert_tdm_to_melotts(self):
        """TDM 데이터를 MeloTTS 형식으로 변환"""
        
        print("🎵 TDM_LLJ → MeloTTS 데이터 변환 시작...")
        
        melotts_data = []
        file_counter = 0
        
        # 언어별 폴더 탐색
        language_folders = []
        
        # 언어 폴더 찾기
        for folder in self.tdm_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name.lower()
                if any(lang in folder_name for lang in ['영어', '일어', '한국어', 'en', 'jp', 'kr']):
                    language_folders.append(folder)
                    print(f"📁 언어 폴더 발견: {folder.name}")
        
        # 언어 폴더가 없으면 루트에서 직접 감정 폴더 찾기
        if not language_folders:
            language_folders = [self.tdm_dir]
        
        for lang_folder in language_folders:
            print(f"\n🔍 처리 중: {lang_folder.name}")
            
            # 감정 폴더 찾기
            emotion_folders = self._find_emotion_folders(lang_folder)
            
            for emotion_folder in emotion_folders:
                emotion_name = self._extract_emotion_from_folder(emotion_folder.name)
                emotion_id = EMOTION_ID_MAP.get(emotion_name, 0)
                
                print(f"  😊 감정 처리: {emotion_folder.name} → {emotion_name} (ID: {emotion_id})")
                
                # PTD 폴더에서 WAV 파일 찾기
                ptd_folder = emotion_folder / "PTD"
                if ptd_folder.exists():
                    wav_files = list(ptd_folder.glob("*.wav"))
                else:
                    wav_files = list(emotion_folder.glob("*.wav"))
                
                print(f"    🎵 WAV 파일 {len(wav_files)}개 발견")
                
                for wav_file in wav_files:
                    segments = self._split_audio_file(wav_file)
                    
                    for i, segment_audio in enumerate(segments):
                        # 새 파일명 생성
                        new_filename = f"{emotion_name}_{file_counter:04d}.wav"
                        new_path = self.output_dir / "wavs" / new_filename
                        
                        # 세그먼트 저장
                        sf.write(new_path, segment_audio, 22050)
                        
                        # 기본 텍스트 생성 (실제로는 ASR이나 수동 라벨링 필요)
                        text = self._generate_sample_text(emotion_name, file_counter)
                        
                        # 메타데이터 추가
                        melotts_data.append({
                            'wav_path': f"wavs/{new_filename}",
                            'speaker_id': 'speaker_00',
                            'language': self.language,
                            'text': text,
                            'emotion_id': emotion_id,
                            'emotion_name': emotion_name,
                            'original_file': str(wav_file)
                        })
                        
                        file_counter += 1
                        
                        if file_counter % 100 == 0:
                            print(f"    ✅ {file_counter}개 세그먼트 처리 완료")
        
        return self._save_melotts_format(melotts_data)
    
    def _find_emotion_folders(self, parent_folder):
        """감정 폴더들을 찾기"""
        emotion_folders = []
        
        for folder in parent_folder.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                # 감정 키워드가 포함된 폴더 찾기
                if any(emotion in folder_name for emotion in EMOTION_MAP.keys()):
                    emotion_folders.append(folder)
        
        return emotion_folders
    
    def _extract_emotion_from_folder(self, folder_name):
        """폴더명에서 감정 추출"""
        folder_lower = folder_name.lower()
        
        # 한국어 감정명 매핑
        for korean_emotion, english_emotion in EMOTION_MAP.items():
            if korean_emotion in folder_name or korean_emotion.lower() in folder_lower:
                return english_emotion
        
        # 기본값
        return 'neutral'
    
    def _split_audio_file(self, wav_file):
        """긴 음성 파일을 세그먼트로 분할"""
        try:
            print(f"      🔪 분할 중: {wav_file.name}")
            
            # 오디오 로드
            audio, sr = librosa.load(wav_file, sr=22050)
            
            # 무음 기반 분할
            audio_segment = AudioSegment.from_wav(wav_file)
            
            # 무음 기준으로 분할 (최소 길이: 3초, 최대 길이: 15초)
            chunks = split_on_silence(
                audio_segment,
                min_silence_len=1000,  # 1초 무음
                silence_thresh=-40,     # -40dB 이하를 무음으로 간주
                keep_silence=500       # 앞뒤 0.5초 무음 유지
            )
            
            # 너무 짧거나 긴 세그먼트 필터링
            filtered_chunks = []
            for chunk in chunks:
                duration = len(chunk) / 1000.0  # 초 단위
                if 3.0 <= duration <= 15.0:  # 3-15초 세그먼트만 유지
                    filtered_chunks.append(chunk)
            
            # AudioSegment를 numpy array로 변환
            segments = []
            for chunk in filtered_chunks[:50]:  # 최대 50개 세그먼트
                # AudioSegment를 WAV 바이트로 변환 후 다시 로드
                chunk_audio = chunk.set_frame_rate(22050).set_channels(1)
                audio_array = chunk_audio.get_array_of_samples()
                audio_np = librosa.util.buf_to_float(audio_array, n_bytes=2)
                segments.append(audio_np)
            
            print(f"        ✂️  {len(segments)}개 세그먼트 생성")
            return segments
            
        except Exception as e:
            print(f"        ❌ 분할 실패: {e}")
            return []
    
    def _generate_sample_text(self, emotion_name, counter):
        """감정에 맞는 샘플 텍스트 생성"""
        
        text_templates = {
            'happy': [
                "오늘 정말 기쁜 하루였어요!",
                "와, 이거 정말 대단하네요!",
                "행복한 순간이네요.",
                "정말 즐거워요!",
                "기쁨이 넘쳐나네요."
            ],
            'sad': [
                "정말 슬픈 일이에요.",
                "마음이 아파요.",
                "눈물이 나네요.",
                "슬픈 하루였어요.",
                "가슴이 먹먹해요."
            ],
            'angry': [
                "정말 화가 나네요!",
                "이건 너무해요!",
                "참을 수 없어요!",
                "화가 치밀어요!",
                "분노가 솟구쳐요!"
            ],
            'depressed': [
                "우울한 기분이에요.",
                "힘든 하루네요.",
                "기운이 없어요.",
                "침울한 마음이에요.",
                "우울증이 느껴져요."
            ],
            'neutral': [
                "안녕하세요.",
                "오늘 날씨가 좋네요.",
                "일반적인 대화입니다.",
                "평범한 하루입니다.",
                "중성적인 톤이에요."
            ]
        }
        
        templates = text_templates.get(emotion_name, text_templates['neutral'])
        return templates[counter % len(templates)]
    
    def _save_melotts_format(self, melotts_data):
        """MeloTTS 형식으로 저장"""
        
        if not melotts_data:
            print("❌ 변환할 데이터가 없습니다!")
            return False
        
        print(f"\n💾 {len(melotts_data)}개 세그먼트를 MeloTTS 형식으로 저장중...")
        
        # 1. metadata.list 생성
        metadata_lines = []
        for data in melotts_data:
            line = f"{data['wav_path']}|{data['speaker_id']}|{data['language']}|{data['text']}|{data['emotion_id']}"
            metadata_lines.append(line)
        
        metadata_file = self.output_dir / "metadata.list"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        # 2. 감정별 통계
        emotion_stats = {}
        for data in melotts_data:
            emotion_name = data['emotion_name']
            emotion_stats[emotion_name] = emotion_stats.get(emotion_name, 0) + 1
        
        # 3. config.json 생성
        config = {
            "data": {
                "training_files": "train.list",
                "validation_files": "val.list",
                "n_emotions": len(EMOTION_ID_MAP),
                "emotion_map": EMOTION_ID_MAP,
                "language": self.language,
                "segment_length": self.segment_length
            },
            "model": {
                "n_emotions": len(EMOTION_ID_MAP),
                "emotion_channels": 256
            }
        }
        
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 4. 변환 로그 저장
        log_data = {
            "total_segments": len(melotts_data),
            "emotion_distribution": emotion_stats,
            "source_files": list(set(data['original_file'] for data in melotts_data))
        }
        
        log_file = self.output_dir / "conversion_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 5. 결과 출력
        print("✅ TDM → MeloTTS 변환 완료!")
        print(f"📁 출력 디렉토리: {self.output_dir}")
        print(f"📊 총 세그먼트: {len(melotts_data)}개")
        print("📈 감정별 분포:")
        
        for emotion, count in sorted(emotion_stats.items()):
            print(f"   {emotion}: {count}개 ({count/len(melotts_data)*100:.1f}%)")
        
        print(f"\n📄 생성된 파일:")
        print(f"   - metadata.list: 메타데이터")
        print(f"   - config.json: 설정 파일")
        print(f"   - conversion_log.json: 변환 로그")
        print(f"   - wavs/: 세그먼트 음성 파일들")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="TDM_LLJ 데이터를 MeloTTS 형식으로 변환")
    parser.add_argument("--tdm_dir", default="MeloTTS/TDM_LLJ", help="TDM_LLJ 디렉토리 경로")
    parser.add_argument("--output_dir", default="./emotion_dataset_tdm", help="출력 디렉토리")
    parser.add_argument("--language", default="KR", help="언어 코드 (KR, EN, JP)")
    parser.add_argument("--segment_length", type=int, default=10, help="세그먼트 길이 (초)")
    
    args = parser.parse_args()
    
    converter = TDMToMeloTTSConverter(
        tdm_dir=args.tdm_dir,
        output_dir=args.output_dir,
        language=args.language,
        segment_length=args.segment_length
    )
    
    success = converter.convert_tdm_to_melotts()
    
    if success:
        print("\n🎯 다음 단계:")
        print("1. 데이터 분할: python split_dataset.py --metadata emotion_dataset_tdm/metadata.list --output_dir emotion_dataset_tdm")
        print("2. 텍스트 전처리: cd MeloTTS/melo && python preprocess_text.py --metadata ../../emotion_dataset_tdm/train.list")
        print("3. 감정 모델 훈련: python train.py --config ../../emotion_dataset_tdm/config.json")
        print("\n💡 참고: 생성된 텍스트는 샘플이므로, 실제 전사(transcription)로 교체하는 것을 권장합니다.")
    else:
        print("❌ 변환 실패")

if __name__ == "__main__":
    main() 