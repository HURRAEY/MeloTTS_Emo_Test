#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applio 전처리 데이터를 MeloTTS 형식으로 변환하는 스크립트
"""

import os
import csv
import json
import shutil
from pathlib import Path
import librosa
from text.cleaner import clean_text_bert
import argparse

# 감정 매핑
EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2, 
    'angry': 3,
    'surprised': 4,
    'fearful': 5,
    'disgusted': 6,
    'excited': 7,
    'disappointed': 8
}

class ApplioToMeloTTSConverter:
    def __init__(self, applio_dir, output_dir, language='KR'):
        """
        Args:
            applio_dir: Applio 출력 디렉토리
            output_dir: MeloTTS 형식 출력 디렉토리  
            language: 언어 코드 (KR, EN, JP 등)
        """
        self.applio_dir = Path(applio_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "wavs").mkdir(exist_ok=True)
        
    def convert_applio_to_melotts(self, metadata_file=None):
        """Applio 데이터를 MeloTTS 형식으로 변환"""
        
        print("🔄 Applio → MeloTTS 데이터 변환 시작...")
        
        # 1. 메타데이터 파일 찾기
        if metadata_file is None:
            metadata_files = [
                self.applio_dir / "metadata.csv",
                self.applio_dir / "filelist.txt", 
                self.applio_dir / "transcripts.csv"
            ]
            metadata_file = next((f for f in metadata_files if f.exists()), None)
            
        if metadata_file is None:
            # 메타데이터 파일이 없으면 디렉토리 구조에서 추출
            return self._extract_from_directory_structure()
        
        # 2. 메타데이터 파일 처리
        return self._process_metadata_file(metadata_file)
    
    def _extract_from_directory_structure(self):
        """디렉토리 구조에서 감정 데이터 추출"""
        
        melotts_data = []
        file_counter = 0
        
        print("📁 디렉토리 구조에서 데이터 추출중...")
        
        # 감정별 디렉토리 탐색
        for emotion_dir in self.applio_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
                
            emotion_name = emotion_dir.name.lower()
            if emotion_name not in EMOTION_MAP:
                print(f"⚠️  알 수 없는 감정: {emotion_name}")
                continue
                
            emotion_id = EMOTION_MAP[emotion_name]
            
            # WAV 파일들 처리
            wav_files = list(emotion_dir.glob("*.wav"))
            print(f"📊 {emotion_name}: {len(wav_files)}개 파일 발견")
            
            for wav_file in wav_files:
                # 파일명에서 텍스트 추출 시도
                text = self._extract_text_from_filename(wav_file.stem)
                
                if not text:
                    text = f"샘플 텍스트 {file_counter}"  # 기본 텍스트
                
                # 새 파일명 생성
                new_filename = f"{emotion_name}_{file_counter:04d}.wav"
                new_path = self.output_dir / "wavs" / new_filename
                
                # 파일 복사
                shutil.copy2(wav_file, new_path)
                
                # 메타데이터 추가
                melotts_data.append({
                    'wav_path': f"wavs/{new_filename}",
                    'speaker_id': 'speaker_00',  # 기본 스피커
                    'language': self.language,
                    'text': text,
                    'emotion_id': emotion_id,
                    'emotion_name': emotion_name
                })
                
                file_counter += 1
        
        return self._save_melotts_format(melotts_data)
    
    def _process_metadata_file(self, metadata_file):
        """메타데이터 파일 처리"""
        
        melotts_data = []
        file_counter = 0
        
        print(f"📄 메타데이터 파일 처리: {metadata_file}")
        
        # CSV 형식 처리
        if metadata_file.suffix == '.csv':
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # 필수 컬럼 확인
                    if not all(k in row for k in ['file_path', 'text']):
                        print(f"⚠️  필수 컬럼 누락: {row}")
                        continue
                    
                    # 감정 정보 추출
                    emotion_name = self._extract_emotion(row)
                    if emotion_name not in EMOTION_MAP:
                        emotion_name = 'neutral'
                    
                    emotion_id = EMOTION_MAP[emotion_name]
                    
                    # 원본 파일 경로
                    orig_path = self.applio_dir / row['file_path']
                    if not orig_path.exists():
                        print(f"⚠️  파일 없음: {orig_path}")
                        continue
                    
                    # 새 파일명 및 경로
                    new_filename = f"{emotion_name}_{file_counter:04d}.wav"
                    new_path = self.output_dir / "wavs" / new_filename
                    
                    # 파일 복사
                    shutil.copy2(orig_path, new_path)
                    
                    # 메타데이터 추가
                    melotts_data.append({
                        'wav_path': f"wavs/{new_filename}",
                        'speaker_id': row.get('speaker_id', 'speaker_00'),
                        'language': self.language,
                        'text': row['text'],
                        'emotion_id': emotion_id,
                        'emotion_name': emotion_name
                    })
                    
                    file_counter += 1
        
        # TXT 형식 처리
        elif metadata_file.suffix == '.txt':
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split('|')
                    if len(parts) < 2:
                        continue
                    
                    file_path, text = parts[0], parts[1]
                    
                    # 감정 추출 (파일명 또는 경로에서)
                    emotion_name = self._extract_emotion_from_path(file_path)
                    if emotion_name not in EMOTION_MAP:
                        emotion_name = 'neutral'
                    
                    emotion_id = EMOTION_MAP[emotion_name]
                    
                    # 파일 처리
                    orig_path = self.applio_dir / file_path
                    if not orig_path.exists():
                        continue
                    
                    new_filename = f"{emotion_name}_{file_counter:04d}.wav"
                    new_path = self.output_dir / "wavs" / new_filename
                    
                    shutil.copy2(orig_path, new_path)
                    
                    melotts_data.append({
                        'wav_path': f"wavs/{new_filename}",
                        'speaker_id': 'speaker_00',
                        'language': self.language,
                        'text': text,
                        'emotion_id': emotion_id,
                        'emotion_name': emotion_name
                    })
                    
                    file_counter += 1
        
        return self._save_melotts_format(melotts_data)
    
    def _extract_emotion(self, row):
        """메타데이터 행에서 감정 추출"""
        # 직접 emotion 컬럼이 있는 경우
        if 'emotion' in row:
            return row['emotion'].lower()
        
        # 파일 경로에서 추출
        if 'file_path' in row:
            return self._extract_emotion_from_path(row['file_path'])
        
        return 'neutral'
    
    def _extract_emotion_from_path(self, file_path):
        """파일 경로에서 감정 추출"""
        path_lower = file_path.lower()
        
        for emotion in EMOTION_MAP.keys():
            if emotion in path_lower:
                return emotion
        
        return 'neutral'
    
    def _extract_text_from_filename(self, filename):
        """파일명에서 텍스트 추출 시도"""
        # 파일명 패턴에 따라 조정 필요
        # 예: speaker1_happy_001_안녕하세요.wav
        parts = filename.split('_')
        if len(parts) > 3:
            return '_'.join(parts[3:])
        return None
    
    def _save_melotts_format(self, melotts_data):
        """MeloTTS 형식으로 저장"""
        
        if not melotts_data:
            print("❌ 변환할 데이터가 없습니다!")
            return False
        
        print(f"💾 {len(melotts_data)}개 파일을 MeloTTS 형식으로 저장중...")
        
        # 1. metadata.list 생성 (MeloTTS 형식)
        metadata_lines = []
        
        for data in melotts_data:
            # MeloTTS metadata 형식: path|speaker|language|text|emotion_id
            line = f"{data['wav_path']}|{data['speaker_id']}|{data['language']}|{data['text']}|{data['emotion_id']}"
            metadata_lines.append(line)
        
        metadata_file = self.output_dir / "metadata.list"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        # 2. 감정별 통계 생성
        emotion_stats = {}
        for data in melotts_data:
            emotion_name = data['emotion_name']
            emotion_stats[emotion_name] = emotion_stats.get(emotion_name, 0) + 1
        
        # 3. config.json 생성
        config = {
            "data": {
                "training_files": "train.list",
                "validation_files": "val.list", 
                "n_emotions": len(EMOTION_MAP),
                "emotion_map": EMOTION_MAP,
                "language": self.language
            },
            "model": {
                "n_emotions": len(EMOTION_MAP),
                "emotion_channels": 256
            }
        }
        
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 4. 결과 요약
        print("✅ 변환 완료!")
        print(f"📁 출력 디렉토리: {self.output_dir}")
        print(f"📊 총 파일 수: {len(melotts_data)}")
        print("📈 감정별 분포:")
        
        for emotion, count in sorted(emotion_stats.items()):
            print(f"   {emotion}: {count}개")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Applio 데이터를 MeloTTS 형식으로 변환")
    parser.add_argument("--applio_dir", required=True, help="Applio 출력 디렉토리")
    parser.add_argument("--output_dir", required=True, help="MeloTTS 형식 출력 디렉토리")
    parser.add_argument("--language", default="KR", help="언어 코드 (KR, EN, JP 등)")
    parser.add_argument("--metadata_file", help="메타데이터 파일 경로 (선택사항)")
    
    args = parser.parse_args()
    
    converter = ApplioToMeloTTSConverter(
        applio_dir=args.applio_dir,
        output_dir=args.output_dir, 
        language=args.language
    )
    
    success = converter.convert_applio_to_melotts(args.metadata_file)
    
    if success:
        print("\n🎯 다음 단계:")
        print("1. 텍스트 전처리: python preprocess_text.py --metadata metadata.list")
        print("2. 학습/검증 분할: python split_dataset.py")
        print("3. 감정 모델 훈련: python train.py --config config.json")
    else:
        print("❌ 변환 실패")

if __name__ == "__main__":
    main() 