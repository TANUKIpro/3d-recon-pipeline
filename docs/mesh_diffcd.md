# mesh_diffcd

## 対象タスク

- Stage 5 (DiffCD): 暗黙表面フィッティング + メッシュ抽出
- 実装: `scripts/stage_diffcd_mesh.py`

## 概要

デノイズ済み点群を NPY に変換し、DiffCD (`fit_implicit.py`) をサブプロセス実行してメッシュ化する。  
PyTorch と JAX の GPU コンテキスト競合を避けるため、DiffCD は別プロセスで実行する設計。

## 入出力関係

前段入力:

- `<output_dir>/object_denoised.ply`

主出力:

- `<output_dir>/object_points.npy` (DiffCD 入力)
- `<output_dir>/diffcd/run_*/...` (DiffCD 生ログ・中間成果物)
- `<output_dir>/object_mesh_raw.ply`
- `<output_dir>/object_mesh.ply` (平滑化後)

後段利用:

- Stage 6 (`texture_bake`) が `<output_dir>/object_mesh.ply` を使用

## 詳細フロー

1. 点群準備:
   - 必要に応じて voxel downsample
   - `object_points.npy` 保存
2. GPU 状況を取得し、auto-tune で `batch_size/n_batches` を調整。
3. DiffCD をサブプロセス実行:
   - `/opt/diffcd/fit_implicit.py`
   - OOM なら safer batch でリトライ
4. 生成ディレクトリから最終メッシュ候補 (`mesh_final_*.ply` など) を探索。
5. `object_mesh_raw.ply` としてコピーし、任意平滑化して `object_mesh.ply` を保存。

## アルゴリズム要点

- Auto-tune:
  - VRAM total/free から batch scale を推定
  - `DIFFCD_AUTO_KEEP_EFFECTIVE_SAMPLES=1` なら `batch * n_batches` をなるべく維持
- OOM retry:
  - baseline と段階比率 (`0.85`, `0.70`, `0.55`) の batch で複数試行
- Smooth:
  - `laplacian` または `taubin` を `trimesh` で適用

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `DIFFCD_BATCH_SIZE` | `5000` | 学習バッチサイズ |
| `DIFFCD_N_BATCHES` | `30000` | 学習バッチ数 |
| `DIFFCD_RESOLUTION` | `512` | 最終メッシュ解像度 (points-per-axis) |
| `DIFFCD_AUTO_TUNE` | `1` | ハードウェアに応じた自動調整ON/OFF |
| `DIFFCD_AUTO_TUNE_RESPECT_MANUAL` | `1` | 手動値指定時の auto-tune 抑止 |
| `DIFFCD_AUTO_KEEP_EFFECTIVE_SAMPLES` | `1` | 有効サンプル量維持 |
| `DIFFCD_AUTO_MIN_N_BATCHES` | `10000` | auto-tune時の `n_batches` 下限 |
| `DIFFCD_AUTO_MIN_BATCH_SCALE` | `0.75` | auto-tune batch scale 下限 |
| `DIFFCD_AUTO_MAX_BATCH_SCALE` | `2.4` | auto-tune batch scale 上限 |
| `DIFFCD_AUTO_SELECT_GPU` | `1` | 複数GPU時に空きVRAM最大GPUを選択 |
| `DIFFCD_GPU_INDEX` | unset | 明示GPU固定 (`CUDA_VISIBLE_DEVICES`) |
| `DIFFCD_XLA_MEM_FRACTION` | auto | JAXメモリ確保率 |
| `JAX_COMPILATION_CACHE_DIR` | `/root/.cache/jax_compilation_cache` | JAXコンパイルキャッシュ |
| `DIFFCD_SMOOTH_METHOD` | `laplacian` | `laplacian` / `taubin` |
| `DIFFCD_SMOOTH_ITERATIONS` | `2` | 平滑化反復数 |
| `DIFFCD_SMOOTH_LAMBDA` | `0.5` | 平滑化係数 |
| `DIFFCD_SMOOTH_TAUBIN_NU` | `-0.53` | Taubin の `nu` |

## 実装上の注意

- `prepare_for_jax()` が `XLA_PYTHON_CLIENT_PREALLOCATE=false` を設定。
- 実際のメッシュ探索は `mesh_final_*.ply` -> `mesh_final.ply` -> `meshes/mesh_*.ply` の順。
- `target_points=1_000_000` (PLY->NPY 変換時) は現時点でコード定数。

## 参考文献

- DiffCD repository: <https://github.com/Linusnie/diffcd>
- JAX GPU memory allocation: <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>
- Trimesh smoothing: <https://trimsh.org/trimesh.smoothing.html>
