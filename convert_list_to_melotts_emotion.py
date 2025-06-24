import os
import sys
import shutil
from jamo import h2j

def split_jamo(text):
    # 한글 자모 분리
    return ' '.join(list(h2j(text)))

def process_line(line):
    parts = line.strip().split('|')
    if len(parts) != 5:
        return None
    wav, speaker, lang, text, emotion_id = parts
    # 경로 수정
    if not wav.startswith('emotion_dataset_tdm/wavs/'):
        wav = os.path.join('emotion_dataset_tdm', wav) if not wav.startswith('wavs/') else os.path.join('emotion_dataset_tdm', wav)
    phones = split_jamo(text.replace(' ', ''))
    phone_tokens = phones.split()
    tone = ' '.join(['0'] * len(phone_tokens))
    word2ph = ' '.join([str(i+1) for i in range(len(phone_tokens))])
    return '|'.join([wav, speaker, lang, text, phones, tone, word2ph, emotion_id])

def convert_file(src, dst):
    with open(src, encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            new_line = process_line(line)
            if new_line:
                fout.write(new_line + '\n')

if __name__ == '__main__':
    for fname in ['train.list', 'val.list']:
        src = os.path.join('emotion_dataset_tdm', fname)
        dst = os.path.join('emotion_dataset_tdm', fname + '.bak2')
        out = os.path.join('emotion_dataset_tdm', fname)
        shutil.copy(src, dst)
        convert_file(dst, out)
    print('경로 변환 및 재포맷 완료!') 