import os

# トレーニングリストのパス
train_list_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list"
output_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list.fixed"

# 言語コードを修正したリストを作成
count_fixed = 0
with open(train_list_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        parts = line.split('|')
        if len(parts) >= 3 and parts[2] == 'JA':
            # JAをJPに置き換え
            parts[2] = 'JP'
            count_fixed += 1
        
        # 修正した行を書き込み
        fixed_line = '|'.join(parts)
        f_out.write(fixed_line)

# 新しいリストで元のリストを置き換え
os.rename(output_path, train_list_path)

print(f"言語コードを修正しました。修正された行数: {count_fixed}") 