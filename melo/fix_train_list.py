import os

# トレーニングリストのパス
train_list_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list"
output_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list.fixed"

# 存在するファイルのみを含む新しいリストを作成
with open(train_list_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        parts = line.split('|')
        if len(parts) < 2:
            continue
        
        # ファイルパスを取得（余分なスペースを削除）
        file_path = parts[0].strip()
        
        # ファイルが存在するか確認
        if os.path.exists(file_path):
            f_out.write(line)
        else:
            print(f"Skipping non-existent file: {file_path}")

# 新しいリストで元のリストを置き換え
os.rename(output_path, train_list_path)

print("Training list has been fixed.") 