# mesh_classical

## 対象タスク

- Stage 5 (Classical): 法線推定 + Screened Poisson + 後処理 + ダウンサンプル
- 実装: `scripts/stage_classical_mesh.py`
- Dashboard では 4 サブタスクを個別承認で進行

## 概要

デノイズ済み点群から Open3D ベースで三角形メッシュを再構成する。  
処理を 4 つのサブタスクに分割し、各段階の中間成果物を保存する。

## 入出力関係

前段入力:

- `<output_dir>/object_denoised.ply`

主出力:

- `<output_dir>/object_mesh_input.ply` (preprocess後点群)
- `<output_dir>/object_points_with_normals.ply`
- `<output_dir>/object_mesh_raw.ply`
- `<output_dir>/object_mesh_postprocessed.ply`
- `<output_dir>/object_mesh.ply` (最終)
- `<output_dir>/object_mesh_preview.ply` (表示用軽量メッシュ)

補助出力:

- `<output_dir>/classical_mesh/` 配下にも同名成果物を保存

後段利用:

- Stage 6 (`mesh_wrap`) が `<output_dir>/object_mesh.ply` を使用
- Stage 7 (`mesh_repair`) が `<output_dir>/object_mesh_wrapped.ply` を使用
- Stage 8 (`texture_bake`) が `<output_dir>/object_mesh_repaired.ply` を使用

## 詳細フロー

1. Preprocess:
   - 点群ロード
   - 条件付き voxel downsample
   - SOR 前処理
2. Main:
   - 法線推定 (`KDTreeSearchParamHybrid`)
   - 法線方向整合
   - Screened Poisson 再構成
   - 密度下位分位トリムと bbox crop
3. Postprocess:
   - 非多様体辺や小連結成分の除去
   - 任意スムージング
4. Downsample:
   - 面数が閾値を超える場合のみ quadric decimation

## アルゴリズム要点

- Poisson: `create_from_point_cloud_poisson`
- 小連結成分除去: `cluster_connected_triangles`
- ダウンサンプル: `simplify_quadric_decimation`
- スムージング: `laplacian` または `taubin` (実装は `stage_diffcd_mesh.smooth_mesh_file` を共用)

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `CLASSICAL_PREPROCESS_ENABLED` | `1` | 前処理有効化 |
| `CLASSICAL_PREPROCESS_VOXEL_RATIO` | `0.003` | 前処理 voxel サイズ比率 (bbox 対角) |
| `CLASSICAL_PREPROCESS_MAX_POINTS` | `700000` | この点数超過時に voxel downsample |
| `CLASSICAL_PREPROCESS_SOR_NEIGHBORS` | `20` | 前処理SOR近傍数 |
| `CLASSICAL_PREPROCESS_SOR_STD_RATIO` | `2.8` | 前処理SOR標準偏差倍率 |
| `POISSON_NORMAL_RADIUS_RATIO` | `0.02` | 法線推定半径比率 |
| `POISSON_NORMAL_MAX_NN` | `32` | 法線推定最大近傍数 |
| `POISSON_NORMAL_ORIENT_K` | `24` | 法線方向整合の近傍数 |
| `POISSON_DEPTH` | `9` | Poisson 深さ |
| `POISSON_SCALE` | `1.08` | Poisson scale |
| `POISSON_LINEAR_FIT` | `0` | Poisson linear fit |
| `POISSON_DENSITY_TRIM_QUANTILE` | `0.02` | 低密度頂点除去分位点 |
| `POISSON_CROP_SCALE` | `1.03` | bbox crop 拡大倍率 |
| `CLASSICAL_POST_MIN_COMPONENT_TRIANGLES` | `400` | 小連結成分除去の最小三角形数 |
| `CLASSICAL_POST_MIN_COMPONENT_RATIO` | `0.01` | 小連結成分除去の最大成分比 |
| `CLASSICAL_AUTO_SMOOTH` | `0` | 後処理スムージング自動適用 |
| `CLASSICAL_SMOOTH_METHOD` | `laplacian` | `laplacian` / `taubin` |
| `CLASSICAL_SMOOTH_ITERATIONS` | `2` | スムージング反復数 |
| `CLASSICAL_SMOOTH_LAMBDA` | `0.5` | スムージング係数 |
| `CLASSICAL_SMOOTH_TAUBIN_NU` | `-0.53` | Taubin の `nu` |
| `CLASSICAL_DOWNSAMPLE_ENABLED` | `1` | 面数ダウンサンプル有効化 |
| `CLASSICAL_DOWNSAMPLE_TARGET_FACES` | `100000` | 目標面数 |
| `CLASSICAL_DOWNSAMPLE_TRIGGER_FACES` | `140000` | 実行トリガ面数 |

補足:

- `CLASSICAL_SMOOTH_METHOD` 未指定時は `DIFFCD_SMOOTH_METHOD` をフォールバック参照。

## 失敗時の典型原因

- 入力点群が空
- Poisson 出力が空メッシュ
- 過度のトリム/クリーニング設定で面が消失

## 参考文献

- Screened Poisson Surface Reconstruction (Kazhdan and Hoppe, 2013): <https://hhoppe.com/screenedpoisson.pdf>
- Open3D Poisson API: <https://www.open3d.org/docs/latest/python_api/open3d.geometry.TriangleMesh.html>
- Garland and Heckbert quadric decimation: <https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf>
