import os

def fix_paths(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line and line.startswith('wavs/'):
                # wavs/ -> emotion_dataset_tdm/wavs/
                line = line.replace('wavs/', 'emotion_dataset_tdm/wavs/', 1)
            f_out.write(line + '\n')

if __name__ == '__main__':
    # train.list 수정
    fix_paths('emotion_dataset_tdm/train.list.bak2', 'emotion_dataset_tdm/train.list')
    # val.list 수정
    fix_paths('emotion_dataset_tdm/val.list.bak2', 'emotion_dataset_tdm/val.list')
    print("경로 수정 완료!") 