import os

def filter_existing_files(input_file, output_file):
    existing_lines = []
    missing_files = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            if len(parts) < 8:
                print(f"잘못된 형식 (라인 {line_num}): {line[:50]}...")
                continue
                
            wav_path = parts[0]
            if os.path.exists(wav_path):
                existing_lines.append(line)
            else:
                missing_files.append(wav_path)
    
    # 필터링된 결과를 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in existing_lines:
            f.write(line + '\n')
    
    print(f"{input_file}: {len(existing_lines)}개 파일이 존재함, {len(missing_files)}개 파일이 누락됨")
    if missing_files:
        print("누락된 파일들:")
        for file in missing_files[:5]:  # 처음 5개만 출력
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... 및 {len(missing_files) - 5}개 더")

if __name__ == '__main__':
    # train.list 필터링
    filter_existing_files('emotion_dataset_tdm/train.list', 'emotion_dataset_tdm/train_filtered.list')
    # val.list 필터링  
    filter_existing_files('emotion_dataset_tdm/val.list', 'emotion_dataset_tdm/val_filtered.list')
    
    # 원본을 백업하고 필터링된 파일로 교체
    os.rename('emotion_dataset_tdm/train.list', 'emotion_dataset_tdm/train.list.bak3')
    os.rename('emotion_dataset_tdm/val.list', 'emotion_dataset_tdm/val.list.bak3')
    os.rename('emotion_dataset_tdm/train_filtered.list', 'emotion_dataset_tdm/train.list')
    os.rename('emotion_dataset_tdm/val_filtered.list', 'emotion_dataset_tdm/val.list')
    
    print("파일 필터링 완료!") 