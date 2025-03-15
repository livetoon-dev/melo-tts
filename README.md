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
