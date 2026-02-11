# contact_hole_repair

## 対象タスク

- Stage 7: Mesh Repair (接地候補穴の局所補修)
- 実装: `scripts/stage_contact_hole_repair.py`

## 概要

`object_mesh_wrapped.ply` から境界ループを検出し、
「下端Y帯域にあり、直径が小さく、下向き法線を持つ」ループだけを補修対象にする。
対象ループは2D投影 + Ear clipping で三角形充填し、補修近傍のみ軽く平滑化する。

## 入出力関係

前段入力:

- `<output_dir>/object_mesh_wrapped.ply`

主出力:

- `<output_dir>/object_mesh_repaired.ply`

補助出力:

- `<output_dir>/contact_hole_repair/object_mesh_repaired.ply`

後段利用:

- Stage 8 (`texture_bake`) が `object_mesh_repaired.ply` を入力

## パラメータ

| 名前 | 既定値 | 説明 |
|---|---:|---|
| `MESH_REPAIR_ENABLED` | `1` | 補修有効化 |
| `MESH_REPAIR_MAX_DIAMETER_RATIO` | `0.08` | 補修対象穴の最大直径 (bbox対角比) |
| `MESH_REPAIR_Y_BAND_RATIO` | `0.06` | 補修対象穴のY帯域 (下端からbbox対角比) |
| `MESH_REPAIR_SMOOTH_ITERS` | `2` | 局所平滑化反復数 |

## 失敗時の挙動

- 候補が見つからない場合は入力メッシュ形状をそのまま保存
- 非多様体化する候補ループはロールバックしてスキップ
- ステージ全体が失敗しない限り、出力メッシュは必ず保存される
