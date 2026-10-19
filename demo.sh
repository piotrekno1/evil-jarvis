#!/usr/bin/env bash
# Creepy Clown end-to-end demo: Polish speech in -> Bielik reply -> Piper voice out.
#
#   ./demo.sh                 talk to the clown, clean deep voice
#   ./demo.sh demon           talk to the clown, metallic-demon voice
#   ./demo.sh text            type instead of talking (clean voice)
#   ./demo.sh text demon      type, metallic-demon voice
#   ./demo.sh silent          talk, replies printed only (no voice)
#
# Anything after "--" is passed straight to creepy-clown, for example:
#   ./demo.sh demon -- --piper-speed 1.15 --input-device 2
#   ./demo.sh -- --prosody plain           (calm, even intonation)
#
# WildEdge telemetry turns on when .env (git-ignored) sets WILDEDGE_DSN.
# Only timings and metadata are sent; upload visitor text or voice with
#   ./demo.sh -- --wildedge-capture text   (or audio)

set -euo pipefail
cd "$(dirname "$0")"

effect="clean"
mode="voice"
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    demon) effect="metallic-demon" ;;
    metallic) effect="metallic" ;;
    clean) effect="clean" ;;
    text) mode="text" ;;
    silent) mode="silent" ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try ./demo.sh --help)"; exit 2 ;;
  esac
  shift
done

missing=0
need() {
  if [[ ! -e "$1" ]]; then
    echo "Missing $1"
    echo "  get it with: $2"
    missing=1
  fi
}
need models/whisper-large-v3-turbo-mlx \
  "uv run --extra models hf download mlx-community/whisper-large-v3-turbo --local-dir models/whisper-large-v3-turbo-mlx"
need models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf \
  "uv run --extra models hf download speakleash/Bielik-4.5B-v3.0-Instruct-GGUF Bielik-4.5B-v3.0-Instruct.Q8_0.gguf --local-dir models"
need models/piper/pl_PL-bass-high.onnx \
  "see the Polish voice section of README.md"
command -v ffmpeg >/dev/null || { echo "Missing ffmpeg: brew install ffmpeg"; missing=1; }
[[ $missing -eq 0 ]] || exit 1

extras=(--extra models --extra polish-voice)
telemetry="off (set WILDEDGE_DSN in .env to send timings to WildEdge)"
if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi
if [[ -n "${WILDEDGE_DSN:-}" ]]; then
  extras+=(--extra telemetry)
  telemetry="WildEdge (timings and metadata only, unless --wildedge-capture is passed)"
fi

args=(--voice-effect "$effect")
case "$mode" in
  text) args+=(--text-only) ;;
  silent) args=(--no-speech) ;;
esac

cat <<EOF

  🎪  Creepy Clown demo
  hearing : $( [[ $mode == text ]] && echo "typed text (Polish)" || echo "Whisper large-v3-turbo (Apple GPU), Polish" )
  thinking: Bielik 4.5B
  voice   : $( [[ $mode == silent ]] && echo "off (replies printed)" || echo "Piper bass, $effect, maniacal intonation" )
  metrics : $telemetry
  input   : $( [[ $mode == text ]] && echo "type a line and press Enter (Ctrl-D quits)" || echo "microphone — stay quiet 2 s for calibration, then speak" )

EOF

exec uv run "${extras[@]}" creepy-clown "${args[@]}" "${extra[@]+"${extra[@]}"}"
