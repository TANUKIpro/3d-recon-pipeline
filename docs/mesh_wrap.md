# mesh_wrap

## 対象タスク

- Stage 6: Mesh Wrap (Iterative Poisson shell)
- Stage 6.5 相当: Contact Hole Repair (接地候補穴の局所補修)
- 実装: `scripts/stage_mesh_wrap.py`
- 実装: `scripts/stage_contact_hole_repair.py`

## 概要

Stage 5 のメッシュを入力に、Poisson 再構成ベースで外皮メッシュを再生成する。  
目的は UV 展開を安定化し、穴や細片で `xatlas` が詰まりやすいケースを緩和すること。

## 入出力関係

前段入力:

- `<output_dir>/object_mesh.ply` (Stage 5 出力)

主出力:

- `<output_dir>/object_mesh_wrapped.ply`
- `<output_dir>/object_mesh_repaired.ply`

補助出力:

- `<output_dir>/mesh_wrap/object_mesh_wrapped.ply`
- `<output_dir>/contact_hole_repair/object_mesh_repaired.ply`

後段利用:

- Stage 7 (`texture_bake`) が `<output_dir>/object_mesh_repaired.ply` を優先使用
- `object_mesh_repaired.ply` が無い場合は `object_mesh_wrapped.ply` を使用

## 詳細フロー

1. 入力メッシュの退化/重複/非多様体を除去。
2. メッシュ表面から点群をサンプリングし、法線を推定・整合。
3. Screened Poisson 再構成で外皮を生成。
4. 低密度頂点トリム + bbox crop + 最大連結成分抽出で整形。
5. 必要に応じて面数を target に再簡略化して保存。

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `MESH_WRAP_ENABLED` | `1` | Wrap ステージ有効化 |
| `MESH_WRAP_METHOD` | `poisson_iterative` | `poisson_iterative` / `ipsr` (`ipsr` は現状フォールバック) |
| `MESH_WRAP_ITERATIONS` | `1` | Poisson wrap 反復回数 |
| `MESH_WRAP_SAMPLE_POINTS` | `180000` | 各反復のサンプル点数 |
| `MESH_WRAP_NORMAL_RADIUS_RATIO` | `0.02` | 法線推定半径比率 (bbox 対角比) |
| `MESH_WRAP_NORMAL_MAX_NN` | `32` | 法線推定近傍上限 |
| `MESH_WRAP_NORMAL_ORIENT_K` | `24` | 法線向き整合近傍数 |
| `MESH_WRAP_POISSON_DEPTH` | `8` | Poisson 深さ |
| `MESH_WRAP_POISSON_SCALE` | `1.05` | Poisson scale |
| `MESH_WRAP_POISSON_LINEAR_FIT` | `0` | Poisson linear fit |
| `MESH_WRAP_DENSITY_TRIM_Q` | `0.02` | 低密度頂点除去分位点 |
| `MESH_WRAP_CROP_SCALE` | `1.03` | bbox crop 拡大倍率 |
| `MESH_WRAP_KEEP_LARGEST_COMPONENT` | `1` | 最大連結成分のみ保持 |
| `MESH_WRAP_TARGET_FACE_RATIO` | `1.10` | 入力面数に対する目標比率 |
| `MESH_WRAP_MIN_FACES` | `25000` | 目標面数の下限 |
| `MESH_WRAP_MAX_FACES` | `120000` | 目標面数の上限 |
| `MESH_WRAP_PRESERVE_INPUT_ON_FAILURE` | `1` | 失敗時に入力メッシュを保存して続行 |
| `CONTACT_HOLE_REPAIR_ENABLED` | `1` | 接地候補穴補修を有効化 |
| `CONTACT_HOLE_MAX_DIAMETER_RATIO` | `0.08` | 補修対象穴の最大直径 (bbox対角比) |
| `CONTACT_HOLE_Y_BAND_RATIO` | `0.06` | 補修対象穴のY帯域 (下端からbbox対角比) |
| `CONTACT_HOLE_SMOOTH_ITERS` | `2` | 補修後の局所平滑反復数 |

## 補足

- 論文 iPSR の完全実装ではなく、現行環境で実行可能な Poisson ベースの近似ラップ。
- `MESH_WRAP_METHOD=ipsr` を指定した場合も、現状は Poisson 方式にフォールバックする。
