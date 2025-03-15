import os

# トレーニングリストのパス
train_list_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list"
output_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list.filtered"

# BERT特徴ファイルが存在するファイルのみを含む新しいリストを作成
valid_count = 0
skipped_count = 0
with open(train_list_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        parts = line.split('|')
        if len(parts) < 1:
            continue
        
        # WAVファイルパスを取得
        wav_path = parts[0].strip()
        # BERTファイルパスを生成
        bert_path = wav_path.replace(".wav", ".bert.pt")
        
        # BERTファイルが存在するか確認
        if os.path.exists(bert_path):
            f_out.write(line)
            valid_count += 1
        else:
            skipped_count += 1
            print(f"スキップしたファイル: {wav_path} (BERT特徴ファイルなし)")

# 新しいリストで元のリストを置き換え
os.rename(output_path, train_list_path)

print(f"BERT特徴ファイルがあるファイルのみを抽出しました。")
print(f"有効なファイル: {valid_count}, スキップしたファイル: {skipped_count}") 