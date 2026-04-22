# extract_frames

## 対象タスク

- Stage 1: フレーム抽出
- 実装: `scripts/stage_extract_frames.py`
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

入力動画から等間隔で JPEG フレームを抽出する前処理タスク。  
動画の回転メタデータを補正し、後段タスクで扱いやすい連番ファイル (`00000.jpg` 形式) を生成する。

## 入出力関係

前段入力:

- `video_path` (`.mp4` 等)

出力:

- `<output_dir>/frames/00000.jpg` から始まる連番JPEG

後段利用:

- Stage 2 (`colmap_sfm`) が `frames/` を直接参照
- Stage 3 (`sam2_segment`) も `frames/` を参照

## 詳細フロー

1. 動画ファイル存在確認、`cv2.VideoCapture` 初期化。
2. FPS を正規化 (`59.94 -> 60` など)。
3. `frame_interval` 未指定時は `fps/2` を既定値として採用。
4. 回転補正角度を検出:
   - `CAP_PROP_ORIENTATION_AUTO`
   - `CAP_PROP_ORIENTATION_META`
   - `ffprobe stream_tags=rotate` (フォールバック)
5. `frame_idx % frame_interval == 0` のフレームを保存。
6. `max_frames` 到達または動画終端で終了。

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `video_path` | 必須 | 入力動画パス |
| `output_dir` | `/data/output` | 出力先ルート |
| `frame_interval` | `None` (実行時自動) | 抽出間隔。未指定なら約 `fps/2` |
| `max_frames` | `None` (実行時自動) | 抽出上限。未指定時は動画長に応じて自動計算 |
| `MAX_FRAMES` (env) | `50` | 動画フレーム総数が取れない場合の最終フォールバック |
| `FRAME_INTERVAL` (env) | `10` | Dashboard 側の設定既定値 (Stage 関数内では直接参照しない) |

既定値の定義元: `scripts/config_defaults.py`

## 補足

- 保存先が既存でも上書き運用前提で動作するため、再実行時は既存成果物管理 (`app.py` の stage reset) に依存する。
- Progress callback は 0-100% の正規化済み値を送る。

## 参考文献

- OpenCV VideoCapture: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>
- FFprobe: <https://ffmpeg.org/ffprobe.html>
