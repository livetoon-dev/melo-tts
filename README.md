<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8133" alt="myshell-ai%2FMeloTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

# MeloTTS 日本語対応強化版

## 主な改良点

- **LINE DistilBERT対応**  
  日本語処理に特化したLINE DistilBERT（768次元）に対応しました。元のモデルでは中国語用のBERT（1024次元）が使用されていましたが、日本語に特化したLINE DistilBERT（768次元）を使用することで、日本語テキストの理解力が向上し、より自然な音声合成が可能になりました。日本語テキストを処理する際には、自動的にLINE DistilBERT用の768次元表現に変換されます。

- **日本語言語コード対応の強化**  
  元のMeloTTSでは「JP」のみが日本語として認識されていましたが、この改良版では言語コードとして「JA」と「JP」の両方に対応しました。国際標準では日本語は「JA」（Japanese）と表記されることが多いため、どちらの表記を使用しても正しく日本語として認識されるようになりました。現在のトレーニングデータでは「JA」を使用しています。

- **エラーハンドリングの改善**  
  未対応の言語コードに対して適切なエラーメッセージを表示するようになりました。以前は明示的なエラーメッセージなしで処理が中断されていましたが、現在は「未対応の言語コード」というメッセージとともに、どの言語コードがサポートされているかが明確に示されます。

## 使い方

1. 環境構築
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

2. 辞書のダウンロード
   ```bash
   cd melo
   uv run python -m unidic download
   uv pip install pyopenjtalk-plus
   ```

3. トレーニング実行
   ```bash
   cd melo
   bash train.sh data/your_config.json 1
   ```
   または、ポート競合を避けるためにランダムポートを使用:
   ```bash
   bash train_random_port.sh data/your_config.json 1
   ```

## データ前処理ガイド

音声合成の品質向上のため、トレーニングデータの前処理を行うことを強くお勧めします。特に、句読点の統一や短いテキストの適切な処理は日本語TTSの品質向上に大きく貢献します。

### データ前処理の重要性

1. **テキスト品質の向上**
   - 半角/全角の統一（「、」「。」「！」「？」など）
   - 三点リーダー（...）の適切な処理
   - 複数の感嘆符/疑問符の正規化

2. **音声品質のフィルタリング**
   - 長すぎる/短すぎる音声の除外
   - 音量の低い音声の除外
   - ファイルエラーのあるデータの除外

### 前処理の実行方法

以下のスクリプトを使用して、データの前処理が可能です：

```bash
cd melo
chmod +x prepare_vtuber_dataset.sh
./prepare_vtuber_dataset.sh
```

このスクリプトは以下の処理を自動的に行います：

1. **テキストクリーニング**：句読点の統一、特殊文字の処理
2. **音声品質チェック**：長さや音量に基づくフィルタリング
3. **BERT特徴量抽出**：LINE DistilBERT特徴量の生成とデータセット分割

各ステップの詳細は次の通りです：

#### 1. テキストクリーニング

`clean_text_data.py`を使用して、テキストの正規化を行います：
- 半角句読点の全角への変換
- 三点リーダーの統一
- 連続する感嘆符/疑問符の制限
- 3文字未満の短いテキストの特定

#### 2. 音声品質チェック

`check_audio_quality.py`を使用して、音声データの品質をチェックします：
- 0.5秒未満の短すぎる音声の除外
- 10秒以上の長すぎる音声の除外
- 音量が極端に小さい音声の除外

#### 3. BERT特徴量抽出

`preprocess_text.py`を使用して、LINE DistilBERT特徴量の抽出とデータセット分割を行います：
- クリーニング済みテキストからBERT特徴量を生成
- トレーニングセットとバリデーションセットの分割
- 設定ファイルの生成

### 前処理のカスタマイズ

各スクリプトはコマンドラインオプションで動作をカスタマイズ可能です：

```bash
# テキストクリーニングのカスタマイズ
python clean_text_data.py --metadata PATH --min-chars 4

# 音声品質チェックのカスタマイズ
python check_audio_quality.py --metadata PATH --min-duration 1.0 --max-duration 8.0
```

## 元のMeloTTSとの差分

- **LINE DistilBERT（768次元）のサポート追加**  
  `data_utils.py`の`get_text`関数で、言語コードに応じてBERTの次元を適切に処理するよう修正しました。日本語テキスト処理時には、LINE DistilBERT互換の768次元表現が使用されます。

- **言語コード「JA」のサポート追加**  
  `data_utils.py`の`get_text`関数で、`language_str in ["JP", "JA", "EN", ...]`のように条件を拡張し、JAコードも正しく処理されるようにしました。

- **エラーハンドリングの強化**  
  未知の言語コードに対して`ValueError`を発生させ、明確なエラーメッセージを表示するようにしました。

- **マルチプロセス設定の最適化**  
  `train.py`と`train.sh`で、PyTorchのマルチプロセス設定とCUDA関連の設定を最適化しました。

- **ポート競合問題の解決**  
  `train_random_port.sh`スクリプトを追加し、ランダムなポート番号を使用してDistributed Trainingの競合を回避できるようにしました。

## レポジトリについて

- **ソースについて**  
  このプロジェクトはMeloTTSからコピーしています。

- **日本語対応の改善**  
  日本語向けに改善を行っています。詳細は[こちら](https://zenn.dev/kun432/scraps/34d9ff1874bd3b)をご参照ください。

- **プロジェクト管理 (uv利用)**  
  プロジェクトは[uv](https://astral.sh/uv)で管理しています。環境構築は以下のコマンドを実行してください:
  
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  ```

- 辞書のダウンロード (トレーニング前に必要)
トレーニング前に、以下のコマンドで辞書をダウンロードしてください:

  ```bash
  cd melo
  uv run python -m unidic download
  uv pip install pyopenjtalk-plus
  ```

- データ配置について
データは melo/data 配下に配置してください。


## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) (project lead) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.

# MeloTTS

MeloTTSは、高品質な音声合成を実現するためのTTS（Text-to-Speech）エンジンです。

## 機能

- 高品質な音声合成
- 複数の言語サポート
- 感情表現の制御
- 高速な推論処理

## インストール

```bash
pip install melo-tts
```

## 使用方法

### 基本的な使用

```python
from melo_tts import MeloTTS

# TTSエンジンの初期化
tts = MeloTTS()

# テキストから音声を生成
text = "こんにちは、世界！"
audio = tts.synthesize(text)
```

### 推論時のBERT特徴量抽出

MeloTTSは推論時にBERT特徴量を効率的に抽出します。以下の機能が利用可能です：

```python
from melo_tts.text import ImprovedJapaneseBertFeatureExtractor

# 特徴量抽出器の初期化
extractor = ImprovedJapaneseBertFeatureExtractor()

# 基本的な特徴量抽出
text = "こんにちは"
word2ph = [1, 1, 1, 1, 1]  # 各トークンが1つの音素に対応
feature = extractor.get_bert_feature(text, word2ph)

# 感情分析による補助テキストを使用
feature = extractor.get_bert_feature(text, word2ph, use_sentiment_assist=True)

# カスタム補助テキストを使用
feature = extractor.get_bert_feature(text, word2ph, assist_text="嬉しい", assist_text_weight=0.7)
```

#### 推論時の特徴

1. **効率的なリソース管理**
   - モデルとトークナイザーは自動的にキャッシュされます
   - メモリ使用量が最適化されます
   - 推論速度が向上します

2. **自動デバイス選択**
   - GPUが利用可能な場合は自動的にGPUを使用
   - Apple Siliconの場合はMPSを利用
   - それ以外の場合はCPUを使用

3. **安定した推論**
   - エラー時も代替特徴量を生成
   - 推論が中断されることはありません
   - 詳細なログ出力で問題追跡が容易

4. **バッチ処理対応**
   - 複数のテキストを一度に処理可能
   - 推論効率が向上

#### 注意点

- 推論時は`torch.no_grad()`が自動的に適用されます
- キャッシュ機能により、メモリ使用量は一定に保たれます
- エラー時は代替特徴量が生成されるため、推論は継続されます
