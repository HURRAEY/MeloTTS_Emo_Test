import json
import sys
import os

# MeloTTS 경로를 추가
sys.path.append('MeloTTS/melo')

# symbols를 import
from text.symbols import symbols

def add_symbols_to_config():
    config_path = 'MeloTTS/melo/configs/config.json'
    
    # 기존 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # data 섹션에 symbols 추가
    config['data']['symbols'] = symbols
    
    # 백업
    backup_path = config_path + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 업데이트된 설정 저장
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"symbols 추가 완료! (총 {len(symbols)}개)")
    print(f"백업 파일: {backup_path}")

if __name__ == '__main__':
    add_symbols_to_config() 