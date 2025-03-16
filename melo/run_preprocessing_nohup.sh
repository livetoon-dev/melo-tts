pkill -f "preprocess_text.py"  # 既存のプロセスを停止
nohup uv run preprocess_text.py \
  --metadata /home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/metadata.list \
  --num-processes 8 \
  --use-sudachi \
  --cleaned-path /home/nagashimadaichi/dev/melo-tts/melo/data/vtuber/metadata.list.cleaned &