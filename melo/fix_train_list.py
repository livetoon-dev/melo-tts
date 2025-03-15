import os

# トレーニングリストのパス
train_list_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list"
output_path = "/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/train.list.fixed"

# 存在するファイルのみを含む新しいリストを作成
with open(train_list_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        # まず行全体を処理
        line_parts = line.split('|')
        if len(line_parts) < 2:
            continue
        
        # 問題：ファイルパスとセパレータの間にスペースがある
        # 例: "/path/to/file.wav |vtuber|..."
        # そのため、最初の部分を特別に処理する
        raw_path = line_parts[0]
        
        # セパレータの前にスペースがあるかチェック
        if ' |' in raw_path:
            # スペースとセパレータの位置を見つける
            space_pos = raw_path.rfind(' |')
            # 実際のファイルパスを取得
            file_path = raw_path[:space_pos].strip()
            # 残りの文字列（通常は空）を取得
            remaining = raw_path[space_pos+1:].strip()
            # 残りの部分が空でない場合、それを次のフィールドの先頭に追加
            if remaining and remaining != '|':
                if len(line_parts) > 1:
                    line_parts[1] = remaining + line_parts[1]
        else:
            file_path = raw_path.strip()
        
        # ファイルが存在するか確認
        if os.path.exists(file_path):
            # 正しいフォーマットで再構築
            corrected_line = file_path + '|' + '|'.join(line_parts[1:])
            f_out.write(corrected_line)
        else:
            print(f"Skipping non-existent file: {file_path}")

# 新しいリストで元のリストを置き換え
os.rename(output_path, train_list_path)

print("Training list has been fixed.") 