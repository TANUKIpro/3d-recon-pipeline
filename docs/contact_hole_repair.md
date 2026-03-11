# contact_hole_repair

## 対象タスク

- Stage 7: Mesh Repair (接地候補穴の局所補修)
- 実装: `scripts/stage_contact_hole_repair.py`

## 概要

Stage 7 は以下の 2 モードを持つ。

1. Dashboard 実行: 境界ループ候補を可視化し、閉曲面閾値スライダで法線 Y フィルタリングを適用、ユーザーがクリック選択したループのみ補修
2. 既存自動実行: ヒューリスティクス（下端Y帯域・直径・法線）で候補を自動選定して補修

補修は 2D 投影 + Ear clipping で三角形充填し、補修近傍のみ軽く平滑化する。

## 入出力関係

前段入力:

- `<output_dir>/object_mesh_wrapped.ply`

主出力:

- `<output_dir>/object_mesh_repaired.ply`

補助出力:

- `<output_dir>/contact_hole_repair/object_mesh_repaired.ply`

後段利用:

- Stage 8 (`texture_bake`) が `object_mesh_repaired.ply` を入力

Dashboard 追加 API:

- `GET /api/mesh-repair/candidates`
- `POST /api/mesh-repair/confirm`

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `MESH_REPAIR_ENABLED` | `1` | 補修有効化 |
| `MESH_REPAIR_MAX_DIAMETER_RATIO` | `0.46` | 補修対象穴の最大直径 (bbox対角比)。接地前提の底面穴を既定で拾えるよう調整済み |
| `MESH_REPAIR_Y_BAND_RATIO` | `0.06` | 補修対象穴のY帯域 (下端からbbox対角比) |
| `MESH_REPAIR_SMOOTH_ITERS` | `3` | 局所平滑化反復数 |

既定値の定義元: `scripts/config_defaults.py`

## Dashboard 閉曲面閾値スライダ

Dashboard の Stage 7 ツールバーに「Closed-surface threshold (normal_y)」スライダを搭載。

- **`normal_y` threshold**: 範囲 `-1.0`〜`0.0`、既定 `-0.25`、ステップ `0.01`
- 閾値以下の `normal_y` を持つ境界ループのみを候補として表示する
- スライダ操作でリアルタイムに候補の表示/非表示が切り替わる
- ユーザーがクリック選択したループは閾値に関わらず常時表示される
- 0 件選択で確定すると補修をスキップし、入力メッシュをそのまま保存する

## 失敗時の挙動

- 候補が見つからない場合は入力メッシュ形状をそのまま保存
- 非多様体化する候補ループはロールバックしてスキップ
- ステージ全体が失敗しない限り、出力メッシュは必ず保存される
