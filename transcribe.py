"""Pre-compute and cache Whisper transcripts for train and test (resumable).

    python src/transcribe.py            # model from GSE_WHISPER_MODEL, default small.en
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_extraction as fx  # noqa: E402

if __name__ == "__main__":
    for split in ("train", "test"):
        df = pd.read_csv(fx.DATA_DIR / f"{split}.csv")
        out = fx.transcribe(df, split)
        if out is None:
            sys.exit("No ASR backend available - install faster-whisper or openai-whisper")
        print(split, "empty transcripts:", int((out["transcript"].str.len() == 0).sum()),
              "errors:", int(out["asr_error"].notna().sum()))
