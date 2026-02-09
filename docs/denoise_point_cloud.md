# denoise_point_cloud

## 対象タスク

- Stage 4: 点群デノイズ
- 実装: `scripts/stage_denoise.py`
- 呼び出し元: `scripts/dashboard/pipeline_runner.py`, `scripts/pipeline.py`

## 概要

`object.ply` からノイズ点を除去し、メッシュ化しやすい点群 `object_denoised.ply` を生成する。  
アルゴリズムをチェーンで選べる構造 (`dbscan`, `sor`, `radius`) になっている。

## 入出力関係

前段入力:

- `<output_dir>/object.ply`

出力:

- `<output_dir>/object_denoised.ply`

後段利用:

- Stage 5 (`mesh_classical` / `mesh_diffcd`) が入力として使用

## 詳細フロー

1. PLY 読み込み (`xyz`, 任意で `rgb`)。
2. 設定解決:
   - `preset` から既定値を読み込む
   - 明示指定されたパラメータで上書き
3. `algorithm` に応じて処理ステップを順次適用:
   - `dbscan_sor` = DBSCAN -> SOR
   - `dbscan_only` = DBSCAN
   - `sor_only` = SOR
   - `radius_only` = Radius outlier
   - `dbscan_radius` = DBSCAN -> Radius outlier
4. 保存 (`object_denoised.ply`)。

## アルゴリズム要点

- DBSCAN: 最大クラスタのみ残す。点数過多時は内部 voxel downsample で近似。
- SOR (Statistical Outlier Removal): k近傍平均距離の統計外れ値除去。
- Radius outlier: 一定半径内近傍数が閾値未満の点を除去。
- `dbscan_eps == 0` のときは `median_bbox_extent * dbscan_eps_ratio` で自動決定。

## パラメータ

| 名前 | 既定値 (balanced) | 説明 |
|---|---:|---|
| `preset` | `balanced` | プリセット名 |
| `algorithm` | `dbscan_sor` | 実行チェーン |
| `dbscan_eps` | `0.0` | DBSCAN半径。0なら自動推定 |
| `dbscan_eps_ratio` | `0.02` | `eps` 自動推定比率 |
| `dbscan_min_samples` | `10` | DBSCAN最小サンプル数 |
| `dbscan_max_points` | `500000` | DBSCAN前の内部ダウンサンプル目標 |
| `sor_neighbors` | `20` | SOR近傍数 |
| `sor_std_ratio` | `2.0` | SOR標準偏差倍率 |
| `radius_neighbors` | `8` | Radius法の最小近傍数 |
| `radius_ratio` | `0.015` | 半径 = `median_bbox_extent * ratio` |

利用可能 `preset`:

- `balanced`
- `detail_preserving`
- `isolate_subject`
- `sparse_noise`
- `aggressive_cleanup`

## 失敗時の典型原因

- 入力点群が空 (`0 points`)。
- 非標準 PLY 形式で `vertex` 属性不足。
- 極端な閾値設定で点群がほぼ消失。

## 参考文献

- DBSCAN (Ester et al., 1996): <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>
- PCL Statistical Outlier Removal: <https://pointclouds.org/documentation/tutorials/statistical_outlier.html>
- PCL Radius Outlier Removal: <https://pointclouds.org/documentation/tutorials/remove_outliers.html>
