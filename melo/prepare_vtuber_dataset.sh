#!/bin/bash

# 環境変数の設定
VENV_PATH="/home/nagashimadaichi/dev/melo-tts/.venv"
DATA_DIR="/home/nagashimadaichi/dev/melo-tts/melo/data/vtuber"
NUM_PROCESSES=8  # 並列処理に使用するプロセス数

# 仮想環境は自動検出されるため、アクティベーションは不要
# source "${VENV_PATH}/bin/activate" 

echo "VTuberデータセット準備開始"

# メタデータファイルの存在確認
if [ ! -f "${DATA_DIR}/metadata.list" ]; then
    echo "エラー: メタデータファイルが見つかりません: ${DATA_DIR}/metadata.list"
    exit 1
fi

# ステップ1: テキストクリーニング
echo "===== ステップ1: テキストクリーニング ====="
echo "テキストクリーニング開始: ${DATA_DIR}/metadata.list"
uv run clean_text_data.py --metadata "${DATA_DIR}/metadata.list"
if [ $? -ne 0 ]; then
    echo "エラー: テキストクリーニングに失敗しました"
    exit 1
fi

# ステップ2: 音声品質チェック
echo "===== ステップ2: 音声品質チェック ====="
echo "音声品質チェック開始: ${DATA_DIR}/metadata.list.txt_cleaned"
uv run check_audio_quality.py --metadata "${DATA_DIR}/metadata.list.txt_cleaned"
if [ $? -ne 0 ]; then
    echo "エラー: 音声品質チェックに失敗しました"
    exit 1
fi

# ステップ3: BERT特徴量抽出とデータセット分割
echo "===== ステップ3: BERT特徴量抽出とデータセット分割 ====="
uv run preprocess_text.py --metadata "${DATA_DIR}/metadata.list.txt_cleaned.audio_filtered" --num-processes ${NUM_PROCESSES}
if [ $? -ne 0 ]; then
    echo "エラー: BERT特徴量抽出に失敗しました"
    exit 1
fi

echo "データセット準備完了！"
echo "トレーニングを開始するには以下のコマンドを実行してください:"
echo "cd /home/nagashimadaichi/dev/melo-tts/melo && bash train_random_port.sh data/vtuber/config.json 1" 