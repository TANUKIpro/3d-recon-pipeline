
π³ / Pi3X と SAM を組み合わせて“物体のマスク付き多視点再構成”を行うための実装部品一式（アーキテクチャ案、データフロー、公式リンク、必要コンポーネント）をまとめる。

---

## 1) ベースラインの全体アーキテクチャ

### A. 入力・データ管理層
- **入力**：単眼RGB動画（mp4 等） or 連番画像
- **フレーム抽出**：一定間隔サンプリング（例：動画なら10フレームに1枚）  
    ※Pi3側も動画入力とフレーム間引きの想定があります。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
- **成果物（中間生成物）**：
    - `frames/`：抽出画像
    - `masks/`：SAMの物体マスク（2値 or RLE）
    - `frames_masked/`：背景を落とした画像（再構成入力）

### B. 物体セグメンテーション層（SAM）
- **SAM（静止画）**：各フレームで「点/Boxプロンプト→マスク」を生成
    - プロンプト入力（点/Box）・自動マスク生成器などが公式で提供されています。 ([GitHub](https://github.com/facebookresearch/segment-anything "GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))
- **SAM 2（動画）**：動画状態（メモリ）を持ってマスクを伝播できるので、同一物体を連続で追う用途に相性が良いです（ベースラインでも採用候補）。 ([GitHub](https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))
- **KV-Tracker論文上の使い方（重要）**：物体追跡では **「SAM等で背景をマスクして、物体中心の画像」を作る**のが前提として述べられています。

### C. 多視点幾何推定・再構成層（Pi3X）
- Pi3Xは入力画像集合から、**カメラ姿勢（camera-to-world）と点群（per-viewの点マップ＋グローバル点群）**を出します。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
    - 出力には `camera_poses`（OpenCV形式4×4）と `points / local_points / conf` が含まれます。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
- Pi3Xは **条件注入（pose / intrinsics / depth）**にも対応し、**近似的なメートルスケール再構成**もサポート、と明記されています（後から拡張しやすい）。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))

### D. 点群融合・出力層（再構成の“形”を作る）
Pi3Xの出力をどう最終モデルにするかで部品が変わります。まずは以下の段階的構成が安全です。
1. **点群の素朴な統合**
    - 各ビューの `local_points` を `camera_poses` でワールドへ変換し、結合
    - `conf`（信頼度）でフィルタ（低い点を落とす） ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
2. **点群の整形**
    - ダウンサンプル（voxel）
    - 外れ値除去（統計・半径）
3. **メッシュ化（必要なら）**
    - Poisson / BPA などでメッシュ抽出（Open3D等の一般的ツールでOK）

---

## 2) データフロー（入出力と依存関係が分かる形）

**入力：RGB動画**  
→ (1) フレーム抽出（間引き）  
→ (2) セグメンテーション（SAM）

- 初期化：1フレーム目で box/点 を与える（or 自動マスク生成→対象選択） ([GitHub](https://github.com/facebookresearch/segment-anything "GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))
- 以降：
    - 静止画SAMなら「前フレームのbboxを流用」等で半自動化
    - SAM2なら「動画 predictor でマスク伝播」が素直 ([GitHub](https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))  
        → (3) 画像前処理
- `masked_image = image * mask + bg_color*(1-mask)`（背景は単色/平均色などが無難）
- 物体bboxでクロップ→正方形パディング→リサイズ（物体が画面を占める比率を上げる）  
    → (4) Pi3X 推論（画像集合をまとめて投入）
- 入力テンソルは `B×N×3×H×W`、値域 `[0,1]` が想定 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
- 出力：`camera_poses`, `points`, `local_points`, `conf` ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))  
    → (5) 点群融合（confフィルタ→統合→整形）  
    → (6) `.ply` 等で保存（Pi3Xの例でも `.ply` 保存が基本導線） ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))

---

## 3) “実装に必要な部品”チェックリスト（集めるもの）

### 必須リポジトリ／公式ページ
- **Pi3 / Pi3X**
    - プロジェクトページ（概要・デモ導線） ([Wang Yifan](https://yyfz.github.io/pi3/ "Pi3"))
    - GitHub（推論方法、入出力仕様、Pi3X更新点） ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))
    - Hugging Face（Pi3X重み） ([Hugging Face](https://huggingface.co/yyfz233/Pi3X?utm_source=chatgpt.com "yyfz233/Pi3X"))
        
- **SAM（Segment Anything）**
    - 公式GitHub（導入、チェックポイント、prompt API） ([GitHub](https://github.com/facebookresearch/segment-anything "GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))
- **（任意だが強く推奨）SAM 2**
    - 公式GitHub（動画追跡API、チェックポイントDL） ([GitHub](https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))

### モデル重み（どれを使うか）
- Pi3X：Hugging Face上の `yyfz233/Pi3X` が公式導線 ([Hugging Face](https://huggingface.co/yyfz233/Pi3X?utm_source=chatgpt.com "yyfz233/Pi3X"))
- SAM：ViT-H/L/B の公式チェックポイントが提供 ([GitHub](https://github.com/facebookresearch/segment-anything "GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))
- SAM2：チェックポイントDLスクリプト＆個別URLが提供 ([GitHub](https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))

### 依存ライブラリ（“コードを書く前に”決める）
- **PyTorch**（Pi3XとSAM/SAM2が前提）
- **OpenCV**（動画→フレーム、前処理、簡易UI）
- **NumPy**（マスク・幾何データ）
- **Open3D**（点群/メッシュ処理：統合・フィルタ・書き出し）
- （任意）**pycocotools**（SAMのRLEマスク運用時） ([GitHub](https://github.com/facebookresearch/segment-anything "GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."))

---

## 4) アーキテクチャ選定のポイント（迷いやすい所だけ先に潰す）
### Pi3X入力の作り方
- 物体再構成では、KV-Trackerの記述通り **背景を落とした“物体中心画像”**が前提として効きます。
- Pi3Xは **信頼度 `conf`** を出すので、融合時のノイズ除去に使う設計が基本になります。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))

### スケール（メートル）とカメラ内参
- Pi3Xは「近似的メートルスケール」や「内参/pose/depth条件注入」をサポートします。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))  
    ベースラインではまず **無条件（RGBのみ）**で動かし、必要になったら
    - 手元キャリブレーション（内参）
    - 既存VIO/SLAMのpose
    - 深度推定  
        を `condition.npz` 形式で注入する方向が拡張しやすいです（フォーマットはリポジトリの `example_mm.py` を参照する導線になっています）。 ([GitHub](https://github.com/yyfz/Pi3?utm_source=chatgpt.com "yyfz/Pi3 - Permutation-Equivariant Visual Geometry Learning"))