"""
_whisper_worker.py — Isolated subprocess for Whisper transcription.

WHY THIS EXISTS:
  Whisper imports PyTorch, which initialises the CUDA runtime inside whatever
  process calls it.  When the torch tensors are later garbage-collected on
  Windows, the CUDA driver sometimes performs cleanup on a GC thread while the
  CUDA context is already partly torn down → C-level access violation → the
  entire process is killed with no Python traceback.

  Running Whisper in a *child* subprocess sidesteps this completely:
    • The child's CUDA context is fully owned by the child.
    • When the child exits, the OS reclaims everything cleanly.
    • The parent process never sees a CUDA object at all.

USAGE (called by VideoAgent._transcribe_to_srt):
  python _whisper_worker.py <audio_path> <srt_path> <model_name>

EXIT CODES:
  0 — success, srt_path has been written
  1 — failure, error written to stderr
"""
import sys
import os

def _sec_to_srt_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    if len(sys.argv) < 4:
        print("Usage: _whisper_worker.py <audio_path> <srt_path> <model_name>",
              file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    srt_path   = sys.argv[2]
    model_name = sys.argv[3]

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import whisper  # type: ignore  — only imported inside the child process
        print(f"[whisper_worker] Loading model '{model_name}'…", flush=True)
        model  = whisper.load_model(model_name)
        print(f"[whisper_worker] Transcribing {audio_path}…", flush=True)
        result = model.transcribe(audio_path, word_timestamps=False)

        segments = result.get("segments", [])
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            lines += [
                str(i),
                f"{_sec_to_srt_time(seg['start'])} --> {_sec_to_srt_time(seg['end'])}",
                seg["text"].strip(),
                "",
            ]

        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        print(f"[whisper_worker] Done — {len(segments)} segments → {srt_path}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[whisper_worker] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
