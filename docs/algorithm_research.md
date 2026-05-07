# RGB動画オブジェクト再構成パイプライン アルゴリズム調査

## 概要

本調査は、現行プロジェクト `clip2mesh` の目的である「RGB動画から、オフラインで高速に、テクスチャ付き3Dオブジェクトを再構成する」ことを前提に、各ステージで使われているアルゴリズムと実装を確認し、より高精度または高速に処理できる候補を評価する。

評価軸は以下とする。

| 評価軸 | 内容 |
|---|---|
| オフライン適性 | 初回ビルド・モデル取得後にネットワークなしで実行できるか |
| 速度 | 現行処理時間を短縮できる見込み |
| 精度 | カメラ姿勢、マスク、メッシュ形状、テクスチャ品質の改善見込み |
| 実装影響 | 既存入出力、Docker依存、UI、後段ステージへの影響 |
| 保守性 | 依存プロジェクトの安定性、ライセンス、既存パッチ量 |

全体として、即時の改善は「既存契約を維持した設定・実装修正」、中期改善は「代替バックエンドを追加して比較」、長期改善は「Stage 2-4をまたぐ再設計」として扱う。

## Stage 1: フレーム抽出

### 現行実装

- 実装: `scripts/stage_extract_frames.py`
- 入力: RGB動画
- 出力: `<output_dir>/frames/00000.jpg` 形式の連番JPEG
- アルゴリズム: OpenCV `VideoCapture.read()` で全フレームを逐次デコードし、`frame_idx % frame_interval == 0` のフレームだけ保存する。回転メタデータは OpenCV と `ffprobe` で補正する。

### ボトルネック

- Pythonループで動画全体を逐次デコードするため、`frame_interval` が大きくても長尺動画では読み飛ばし効果が限定的。
- 等間隔抽出のみのため、ブレ、重複、視差不足のフレームが後段のSfMやセグメンテーションに流れる。
- 再構成に重要な「視点変化があるがブレていないフレーム」を選ぶ仕組みがない。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| 現行OpenCV逐次抽出 | 可 | 中 | 中 | 低 | 小規模動画では十分。最も安定 |
| FFmpeg `select` filter 抽出 | 可 | 高 | 中 | 低-中 | 長尺動画で高速化しやすい。出力契約を維持可能 |
| FFmpeg抽出 + ブレ/重複除外 | 可 | 中-高 | 高 | 中 | 後段SfM成功率の改善が期待できる |
| キーフレーム/光学フローによる視差選別 | 可 | 中 | 高 | 中-高 | 品質は上がるが、実装と評価がやや重い |

### 結論

短期は現行OpenCV実装を残しつつ、`ffmpeg` ベースの抽出モードを追加するのが妥当。出力形式を `frames/00000.jpg` のまま維持できるため、Stage 2以降への影響は小さい。

精度改善としては、等間隔抽出の後に Laplacian variance によるブレ除外、ヒストグラム差分や特徴点マッチ数による重複除外を追加する。まずは候補フレームを `max_frames` より多めに抽出し、品質スコアで上位を残す方式が安全である。

根拠:

- FFmpeg公式ドキュメント: <https://ffmpeg.org/ffmpeg.html>
- FFmpeg select filter: <https://ffmpeg.org/ffmpeg-filters.html>
- OpenCV VideoCapture: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>

## Stage 2: COLMAP SfM

### 現行実装

- 実装: `scripts/stage_colmap_sfm.py`
- 入力: `frames/*.jpg`
- 出力: `camera_poses.json`, `intrinsics.json`, `colmap_sparse/`, `colmap_sparse_points.ply`
- アルゴリズム: COLMAP `feature_extractor` でSIFT特徴抽出、`exhaustive_matcher` または `sequential_matcher` で特徴マッチング、`mapper` でインクリメンタルSfMを実行する。`cameras.bin` から fx/fy/cx/cy に加えて歪み係数 (SIMPLE_RADIAL の k1 等) を OpenCV 形式 `[k1, k2, p1, p2, k3]` に変換して `intrinsics.json` に書き出し、Stage 5 がフレームをアンディストーションして使えるようにする。

現行設定は、`scripts/config_defaults.py` では `exhaustive`, `32768 features`, `image_size=2048`, `DSP-SIFT=true`, `GPU=false` が既定である。一方、`docker-compose.yml` では `COLMAP_MAX_FEATURES=8192` が指定されており、設定の意図が少し分散している。

### ボトルネック

- `exhaustive_matcher` は画像枚数に対してペア数がO(N^2)で増える。動画入力では隣接フレームの重なりが大きいため、全ペア照合は無駄が多い。
- 高特徴点数、DSP-SIFT、CPU実行の組み合わせは堅牢だが、オフライン高速目標とは相性が悪い。
- SfM失敗時の自動リトライ設定がないため、速度優先と堅牢性優先を運用で切り替える必要がある。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| 現行SIFT + exhaustive | 可 | 低 | 高 | 低 | 精度重視のfallbackとして維持 |
| SIFT + sequential + GPU | 可 | 高 | 中-高 | 低 | 動画入力の標準候補。短期の本命 |
| sequential失敗時にexhaustiveへ自動fallback | 可 | 高 | 高 | 中 | 速度と成功率の両立が可能 |
| COLMAP 4系 ALIKED + LightGlue | 可 | 高 | 高 | 高 | ONNX有効ビルドが必要。中期候補 |
| hloc + LightGlue | 可 | 高 | 高 | 高 | 高性能だがCOLMAP database連携と依存が増える |
| DUSt3R / MASt3R / Fast3R系 | 重み同梱後は可 | 高 | 高 | 非常に高 | Stage 2-4再設計候補。現行置換には重い |

### 結論

短期の推奨は、動画入力向けの `fast_offline` プリセットを追加し、`sequential matcher`, `GPU on`, `8192-12000 features`, `DSP-SIFT off` を既定に寄せること。再構成に失敗した場合のみ `exhaustive`, `DSP-SIFT on`, 高特徴数の `robust` プリセットで自動リトライする。

中期では、COLMAP 4系のALIKED/LightGlue対応を検証する。COLMAP公式ドキュメントでは、ALIKED特徴抽出とLightGlueマッチングがコマンドラインから選択可能になっている。ただしUbuntu 22.04のapt版COLMAPでは利用できない可能性が高く、ONNX有効ビルドをDockerへ組み込む必要がある。

DUSt3R、MASt3R、Fast3Rは、COLMAPが失敗する低テクスチャ・少数視点環境で有望だが、現行Stage 4が `colmap_sparse/` を前提にしているため、まずは研究用の別パイプラインとして評価する。

根拠:

- COLMAP tutorial: <https://colmap.readthedocs.io/en/latest/tutorial.html>
- COLMAP feature extraction and matching: <https://colmap.github.io/features.html>
- LightGlue公式repo: <https://github.com/cvg/LightGlue>
- DUSt3R公式repo: <https://github.com/naver/dust3r>
- Fast3R公式repo: <https://github.com/facebookresearch/fast3r>

## Stage 3: SAM2セグメンテーション

### 現行実装

- 実装: `scripts/stage_sam2_ui.py`, `scripts/dashboard/sam2_service.py`
- 入力: `frames/*.jpg`
- 出力: `masks/`, `masks_object_raw/`, `masks_ground/`
- アルゴリズム: SAM2 video predictorを使い、ユーザークリックで対象物体とground/contact surfaceを指定し、全フレームへマスクを伝播する。最終マスクは `object_raw AND NOT ground_raw`。

設定上は `SAM2_MODEL=tiny/small/base/large` が存在するが、`SAM2Session._load_model()` は `SAM2_MODEL_CONFIGS["large"]` を固定使用している。そのため、UIや環境変数で軽量モデルを選んでも実体はlargeになる。

### ボトルネック

- 常にlargeモデルをロードするため、SAM2.1 tiny/smallによる高速化が使えていない。
- `offload_video_to_cpu=True`, `offload_state_to_cpu=True` はVRAM節約には有効だが、十分なVRAMがある環境ではCPU-GPU転送が遅くなる可能性がある。
- Stage 4/5の品質はマスク境界に強く依存するため、軽量化時にはマスク漏れ・欠けの検証が必要。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| SAM2.1 large継続 | 可 | 中 | 高 | 低 | 高品質基準として維持 |
| SAM2.1 small/tinyを有効化 | 可 | 高 | 中-高 | 低 | 短期の本命。現行バグ修正で効果が出る |
| VRAM別offload設定 | 可 | 中-高 | 同等 | 低-中 | 16GB以上なら高速化余地あり |
| EdgeTAM | 可 | 高 | 中-高 | 中-高 | SAM2系軽量VOS候補。API検証が必要 |
| MobileSAM/EdgeSAM系 | 可 | 高 | 中 | 中-高 | 静止画向けが中心で、動画伝播は別実装が必要 |

### 結論

最優先は、`SAM2Session._load_model()` で `self.model_type` を使うよう修正し、SAM2.1 tiny/small/base/largeを実際に選択可能にすること。公式ベンチではSAM2.1 tinyはlargeより大幅に高速で、SA-V J&Fの差は限定的である。対象物体はUIで確認・Redoできるため、デフォルトは `small`、速度優先プリセットは `tiny` が妥当である。

次に、VRAMティアに応じて `offload_video_to_cpu` と `offload_state_to_cpu` を切り替える。低VRAMでは現行通りCPU offload、高VRAMではGPU保持を選べるようにする。

EdgeTAMは高速化候補だが、SAM2と同じ入出力契約で置き換えられるかを確認してから比較用バックエンドとして扱う。

根拠:

- SAM2公式repo: <https://github.com/facebookresearch/sam2>
- SAM2論文: <https://arxiv.org/abs/2408.00714>
- EdgeTAM公式repo: <https://github.com/facebookresearch/EdgeTAM>

## Stage 4: gs2mesh再構成

### 現行実装

- 実装: `scripts/stage_gs2mesh_reconstruct.py`, `scripts/gpu_tsdf.py`, `scripts/gs2mesh_config.py`
- 入力: `frames/`, `colmap_sparse/`, `masks/`
- 出力: `object_mesh.ply`
- アルゴリズム:
  1. COLMAP `image_undistorter` で画像を無歪みに変換。
  2. graphdeco-inria版 3D Gaussian Splatting を学習。
  3. gs2meshの `run_single.py` で合成ステレオビューを生成し、DLNRで深度推定。
  4. SAM2マスクを `left_mask.npy` として各ビューに反映。
  5. Open3D `VoxelBlockGrid` でTSDF統合。
  6. Marching Cubesでメッシュ抽出し、小クラスタ除去と面向き補正を行う。

現行は既にgs2mesh標準のCPU TSDFを使わず、Open3Dのtensor-based TSDFへ置き換えているため、同系統の実装としてはかなり高速化されている。

### ボトルネック

- 3DGS学習がStage 4内で最も重い。README上の目安では5k iterationsで3-5分級。
- gs2meshとgraphdeco-inria版gaussian-splattingのAPI互換パッチがDockerfile内に多く、保守負荷が高い。
- DLNR深度推定は精度が高い一方、ビューごとのレンダリングと推論が必要。
- GPU TSDFは6-8GB級のVRAMを使い、低VRAM環境ではCPU TSDFにfallbackする。
- マスクerosion/closingや深度閾値により、薄物や細部が削れる可能性がある。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| 現行gs2mesh + Open3D TSDF | 可 | 中-高 | 高 | 低 | 基準線として維持 |
| 3DGS学習バックエンドをgsplat化 | 可 | 高 | 同等 | 中-高 | 高速化・VRAM削減の有力候補 |
| 2D Gaussian Splatting | 可 | 中 | 高 | 高 | 幾何精度改善候補。別エンジン向き |
| Gaussian Opacity Fields | 可 | 中 | 高 | 高 | surface extractionに強いが移植負荷大 |
| SuGaR | 可 | 中-高 | 中-高 | 中-高 | 3DGSから高速mesh抽出。比較候補 |
| COLMAP dense PatchMatch + fusion | 可 | 中 | 中 | 中 | 古典MVS fallback。主力置換ではない |
| MASt3R-SLAM / Fast3R系 | 重み同梱後は可 | 高 | 高 | 非常に高 | Stage 2-4再設計候補 |

### 候補別評価

#### gsplat

`gsplat` は3DGSの高速・省メモリ実装として有力である。現行のgraphdeco実装を直接置き換えると、gs2mesh側のrenderer互換が問題になる可能性があるため、まずは実験分岐として3DGS学習時間、VRAM、出力point cloud互換性を評価する。

短期の本命は「現行Stage 4の外部契約を維持しつつ、3DGS学習だけをgsplatで置換できるか」を確認することである。

#### 2D Gaussian Splatting

2DGSは3D Gaussianではなく2D oriented diskで面を表現し、深度歪みや法線整合の正則化により幾何的に整ったsurface reconstructionを狙う。現行gs2meshが「3DGSから合成深度を作ってTSDFへ入れる」のに対し、2DGSは表面表現そのものを改善する方向である。

薄物、曲面、面の整合性では改善余地があるが、Stage 4の別エンジンとして `object_mesh.ply` を出すアダプタが必要になる。

#### SuGaR / GOF

SuGaRは3DGSからSurface-Aligned Gaussianを使ってmesh extractionを行う。既存3DGS資産を活用できる可能性があるが、gs2meshの論文・設計思想では、in-the-wild動画から滑らかな面を得るためにステレオ深度を使う点が強みである。したがって、SuGaRは即置換ではなく比較候補とする。

GOFはGaussian Opacity Fieldからlevel setを抽出するsurface reconstruction候補で、幾何品質は期待できるが、現行のSAM2マスク、TSDF、Stage 5との接続を再設計する必要がある。

### 結論

短期は現行gs2mesh + Open3D TSDFを維持し、基準計測を整える。高速化の第一候補は `gsplat` 互換検証である。成功すれば、Stage 4の主ボトルネックである3DGS学習時間とVRAM使用量を下げられる可能性がある。

精度改善の本命は2DGSをStage 4の別エンジンとして追加し、同一入力から `object_mesh.ply` を生成して比較すること。現行を壊さず、メッシュ品質、薄物保持、Stage 5成功率を比較できる。

GOF、MASt3R-SLAM、Fast3R系は有望だが、現行Stage 4の差し替えではなく、Stage 2-4をまたぐ研究枠として扱う。

根拠:

- gs2mesh公式repo: <https://github.com/yanivw12/gs2mesh>
- gs2mesh project page: <https://gs2mesh.github.io/>
- Open3D VoxelBlockGrid: <https://www.open3d.org/docs/release/python_api/open3d.t.geometry.VoxelBlockGrid.html>
- gsplat docs: <https://docs.gsplat.studio/main/>
- 2D Gaussian Splatting公式repo: <https://github.com/hbb1/2d-gaussian-splatting>
- SuGaR公式repo: <https://github.com/Anttwo/SuGaR>
- Gaussian Opacity Fields公式repo: <https://github.com/autonomousvision/gaussian-opacity-fields>

## Stage 5: テクスチャベイク

### 現行実装

- 実装: `scripts/stage_texture_bake.py`, `scripts/texture/`
- 入力: `object_mesh.ply`, `camera_poses.json`, `intrinsics.json`, `frames/`, `masks/`
- 出力: `textured_mesh.obj`, `textured_mesh.mtl`, `texture.png`
- アルゴリズム:
  1. メッシュ、カメラ姿勢、内部パラメータ (歪み係数を含む) を読み込む。
  2. `intrinsics.dist_coeffs` が非ゼロなら `cv2.initUndistortRectifyMap` でアンディストーションマップを構築し、フレーム/マスクを `cv2.remap` で歪み補正する。COLMAP の bundle adjustment は SIMPLE_RADIAL 前提で世界座標を解いているため、ピンホール投影で歪み画像をサンプルすると曲面上で詳細が消える。
  3. `xatlas` でUV展開する。大規模メッシュでは空間分割して並列UV生成する。
  4. 各テクセルについて、法線角度、距離、可視性、SAM2マスクを使いTop-Kビューをスコアリングする。
  5. `region_gc` では競合領域をregion化し、single-view固定で境界破綻を抑える。
  6. non-conflict領域はTop-Kブレンド、未充填領域はfallback scanで埋める。
  7. seam leveling、必要に応じたquality boost、seam padding、sharpenを適用する。

現行は、nvdiffrastによるGPU深度ラスタライズ、LRUキャッシュ、並列UV展開、COLMAP intrinsics直接利用 (歪み係数も含む) など、既に複数の高速化が入っている。

### ボトルネック

- 主要コストは「有効テクセル数 × ビュー数」のTop-K view scoringである。
- GPU深度ラスタライズを使っても、viewごとにCPUへ戻す処理や、フレーム/マスク/深度のCPU側判定が残る。
- `TEXTURE_QUALITY_BOOST` は境界品質を上げるが、ECC/phase correlation等の追加処理により重くなる。
- 全ビュー評価は品質には強いが、動画由来の連続フレームでは冗長なビューが多い。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| 現行region_gc継続 | 可 | 中 | 高 | 低 | 基準線として維持 |
| GPU常駐スコアリング/カラーサンプリング | 可 | 高 | 同等 | 中 | 短期-中期の本命 |
| face/view sparse preselection | 可 | 高 | 同等-高 | 中 | 冗長ビュー削減に有効 |
| MVS-Texturing | 可 | 中-高 | 高 | 高 | 実績あり。別バックエンド向き |
| OpenMVS TextureMesh | 可 | 中-高 | 高 | 高 | カメラ形式変換とmask統合が必要 |
| AliceVision Texturing | 可 | 中 | 高 | 高 | 高品質だが依存と統合が重い |
| Diffusion系テクスチャ生成 | モデル同梱後は可 | 低-中 | 用途次第 | 高 | 入力画像忠実性が目的なので主力外 |

### 結論

Stage 5は全置換ではなく、現行方式をベースに高速化するのが最も現実的である。第一候補は、投影、mask判定、depth判定、bilinear color samplingをTorch tensor化し、GPU上で完結させること。nvdiffrastを既に導入しているため、GPU処理への寄せ方は既存設計と整合する。

第二候補は、face centroidやface normalを使って候補ビューを先に絞る `face/view sparse preselection` である。これにより計算量を `texels × all_views` から `texels × candidate_views` に落とせる。動画入力では隣接フレームが冗長なため、品質低下を抑えながら高速化できる可能性が高い。

MVS-Texturing、OpenMVS、AliceVisionは高品質な既存テクスチャリング実装だが、カメラ形式、メッシュ形式、SAM2マスク、現行のregion_gc品質補正との統合が重い。比較用バックエンドとして小さく検証する位置付けが妥当である。

根拠:

- nvdiffrast: <https://nvlabs.github.io/nvdiffrast/>
- xatlas公式repo: <https://github.com/jpcy/xatlas>
- MVS-Texturing公式repo: <https://github.com/nmoehrle/mvs-texturing>
- OpenMVS: <https://github.com/cdcseacave/openMVS>
- AliceVision Texturing: <https://meshroom.readthedocs.io/en/stable/generated/meshroom.nodes.aliceVision.Texturing.Texturing.html>

## Stage 6: Post-texture Contact Cleanup

### 現行実装

- 実装: `scripts/stage_post_texture_contact_cleanup.py`, `scripts/ground_plane_extraction.py`, `scripts/repair/`
- 入力: `textured_mesh.obj`, `camera_poses.json`, `intrinsics.json`, `masks/`, `masks_ground/`, `ground_plane.json`, `object_mesh.ply`
- 出力: `<object_name>/textured_mesh_cleaned.obj`, `texture.png`, 必要に応じて `texture_cap.png`
- アルゴリズム:
  1. ground maskとメッシュから接地平面を推定する。
  2. faceごとにground mask投影一致率、object mask投影一致率、接地平面距離、近傍状態を評価する。
  3. 複数パスで除去候補を収束させる。
  4. ユーザーがApplyした場合、接地平面でclipし、底面をcap生成して最終OBJを出力する。

### ボトルネック

- faceごとに複数ビューへ投影してmask scoreを計算するため、face数とview数に比例して重くなる。
- centroid一点投影のため、細長いfaceや大きいfaceでは誤判定が起こり得る。
- 独自のclip/cap生成は軽量だが、複雑な境界ループや自己交差に弱い可能性がある。
- Stage 6は人手レビューを挟むため、完全自動の過剰除去は避ける必要がある。

### 候補手法

| 候補 | オフライン | 速度 | 精度 | 実装影響 | 評価 |
|---|---:|---:|---:|---:|---|
| 現行cleanup継続 | 可 | 中 | 中 | 低 | 基準線として維持 |
| face候補限定 + projection/mask cache | 可 | 高 | 同等 | 低-中 | 短期の本命 |
| face内複数サンプル判定 | 可 | 中 | 高 | 中 | centroid誤判定を減らす |
| Open3D RaycastingScene | 可 | 中 | 高 | 中 | 可視性・距離判定の補強に有効 |
| CGAL Polygon Mesh Processing | 可 | 中 | 高 | 高 | 幾何処理は堅牢だが依存とライセンス確認が重い |
| Blender/bmesh外部処理 | 可 | 低-中 | 中-高 | 高 | Docker肥大化と運用負荷が大きい |

### 結論

短期は、接地平面距離とbbox下部判定で候補faceを絞ってからmask投影を行い、pose行列、mask画像、投影行列をキャッシュする。これは出力仕様を変えずに高速化できる。

精度改善として、除去境界付近だけface内複数サンプルを使う。全faceを高密度サンプリングすると重くなるため、現行centroid判定で曖昧なfaceに限定する。

Open3D RaycastingSceneは、既にOpen3Dを依存に持つ現行プロジェクトと相性がよい。可視性や最近傍距離を使って、mask投影だけでは判定しにくい誤除去を減らす候補である。

CGAL相当のclip/hole fillは最も堅牢だが、C++依存、Dockerビルド、ライセンス確認が必要になる。導入する場合は、Stage 6全体の置換ではなく、cap生成部分だけを外部バックエンドとして検証する。

根拠:

- Open3D RaycastingScene: <https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html>
- CGAL Polygon Mesh Processing: <https://doc.cgal.org/latest/Polygon_mesh_processing/index.html>
- Open3D geometry processing docs: <https://www.open3d.org/docs/latest/>

## 総合判断

### フェーズ別結論

| Stage | 結論 | 優先度 |
|---|---|---:|
| Stage 1 フレーム抽出 | FFmpeg抽出モードと簡易品質選別を追加する。現行OpenCVはfallbackとして維持 | 中 |
| Stage 2 COLMAP SfM | `sequential + GPU + 適正特徴数` を高速既定にし、失敗時だけrobust設定へfallback | 高 |
| Stage 3 SAM2 | large固定バグを修正し、SAM2.1 small/tinyを実際に使えるようにする | 高 |
| Stage 4 再構成 | 現行gs2meshを基準線に維持し、`gsplat` と2DGSを比較候補として追加検証 | 高 |
| Stage 5 テクスチャ | 現行region_gcを維持し、GPU常駐化と候補ビュー削減で高速化 | 中-高 |
| Stage 6 cleanup | 候補face限定、projection/mask cache、曖昧faceの複数サンプル化を優先 | 中 |

### 最短で効果が出る改善

1. Stage 3のSAM2モデル選択バグ修正。
2. Stage 2の高速プリセットとrobust fallback導入。
3. Stage 5の候補ビュー削減。
4. Stage 6のprojection/mask cache。
5. Stage 1のFFmpeg抽出モード。

これらは既存の主要出力契約を維持でき、後段ステージへの影響が限定的である。

### 中期の比較候補

1. COLMAP 4系 ALIKED/LightGlue。
2. Stage 4の `gsplat` 学習バックエンド。
3. Stage 4の2DGS別エンジン。
4. Stage 5のMVS-Texturing/OpenMVSバックエンド。
5. Stage 6のOpen3D RaycastingScene補強。

中期候補は、既存パイプラインを直接置換せず、設定で選べる比較バックエンドとして追加するのが安全である。

### 長期の研究候補

- MASt3R-SLAM、Fast3R、DUSt3R系を使ったCOLMAP代替。
- Gaussian Opacity Fieldsによるsurface reconstruction。
- Stage 2-4をまたぐpose/depth/mesh統合パイプライン。

これらは精度・速度の伸びしろが大きいが、現行の `colmap_sparse/`, `camera_poses.json`, `object_mesh.ply` 契約と大きく異なるため、主パイプラインとは別に評価する。

## 実行ロードマップ

### Step 1: ベースライン計測

現行 `default` と `high` プリセットで、以下を記録する。

- 総処理時間
- Stage別処理時間
- GPUピークVRAM
- COLMAP登録フレーム数
- Stage 4の頂点数、三角形数、連結成分数
- Stage 5のテクスチャ未充填率
- Stage 6の除去候補face数
- 最終OBJの目視品質

### Step 2: 低リスク改善

- Stage 3: `SAM2_MODEL` が実際に反映されるようにする。
- Stage 2: `fast_offline` と `robust` の2プリセットを追加し、自動fallbackする。
- Stage 6: mask/pose/projection cacheを追加する。

### Step 3: 中リスク高速化

- Stage 1: FFmpeg抽出モードを追加する。
- Stage 5: face/view sparse preselectionを追加する。
- Stage 5: GPU常駐スコアリングを実験実装する。

### Step 4: 代替バックエンド比較

- Stage 4: `gsplat` 学習バックエンドの互換性を検証する。
- Stage 4: 2DGSから `object_mesh.ply` を生成する別エンジンを検証する。
- Stage 5: MVS-Texturing/OpenMVSを比較用バックエンドとして検証する。

## 受け入れ基準

高速化候補は、同一入力に対して以下を満たす場合に採用する。

- 最終出力のOBJ/MTL/PNG契約を壊さない。
- 現行よりStage単体時間が20%以上短縮する、またはOOM率が明確に下がる。
- 目視品質が現行同等以上である。
- マスク外漏れ、穴、薄物欠落、テクスチャseamが悪化しない。
- 既存テストに加え、少なくとも1つの実動画fixtureでE2E確認できる。

精度改善候補は、時間が増える場合でも以下を満たすなら採用価値がある。

- 薄物、曲面、接地部、低テクスチャ面のいずれかで明確な品質改善がある。
- 既定ではなく `high_quality` または実験プリセットとして選択可能。
- 依存ライセンスとモデルライセンスがプロジェクト用途に合う。

## 注意点

- 現行依存には、3D Gaussian Splattingやnvdiffrastなど、商用利用に制限がある可能性のあるコンポーネントが含まれる。代替手法を追加する際も、モデル重みとコードのライセンスを個別に確認する。
- Web検索で見つかる最新手法には、論文上の性能は高くても、公式コード未公開、ライセンス不明、重い学習環境が必要なものがある。主パイプラインへ入れる前に、Dockerビルド再現性とoffline実行性を確認する。
- 「RGB動画から単一オブジェクトを再構成する」という目的では、汎用scene reconstructionの高性能手法がそのまま最適とは限らない。SAM2マスク、接地面処理、テクスチャ品質まで含めたE2E評価を優先する。
