import os

def fix_to_absolute_paths(input_file, output_file):
    base_dir = os.path.abspath('.')  # 현재 디렉터리의 절대 경로
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            if len(parts) >= 8:
                # 첫 번째 부분(wav 경로)을 절대 경로로 변환
                wav_path = parts[0]
                if not os.path.isabs(wav_path):
                    abs_wav_path = os.path.join(base_dir, wav_path)
                    parts[0] = abs_wav_path
                
                line = '|'.join(parts)
            
            f_out.write(line + '\n')

if __name__ == '__main__':
    # train.list를 절대 경로로 변환
    fix_to_absolute_paths('emotion_dataset_tdm/train.list', 'emotion_dataset_tdm/train_abs.list')
    # val.list를 절대 경로로 변환
    fix_to_absolute_paths('emotion_dataset_tdm/val.list', 'emotion_dataset_tdm/val_abs.list')
    
    # 백업하고 교체
    os.rename('emotion_dataset_tdm/train.list', 'emotion_dataset_tdm/train.list.bak4')
    os.rename('emotion_dataset_tdm/val.list', 'emotion_dataset_tdm/val.list.bak4') 
    os.rename('emotion_dataset_tdm/train_abs.list', 'emotion_dataset_tdm/train.list')
    os.rename('emotion_dataset_tdm/val_abs.list', 'emotion_dataset_tdm/val.list')
    
    print("절대 경로로 변환 완료!")
    
    # 첫 번째 줄 확인
    with open('emotion_dataset_tdm/train.list', 'r') as f:
        first_line = f.readline().strip()
        print(f"첫 번째 라인: {first_line[:100]}...") 