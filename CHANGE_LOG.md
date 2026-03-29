# Change Log

## 2026-03-29

### Real-world bimanual HIL-SERL bring-up

#### Summary

LeRobot 上で、`bi_so_leader + keyboard` を使った bimanual 実機 HIL-SERL を開始できる土台を追加した。
今回の目的は、最初に動く学習パイプラインを作ることと、今後も upstream を追従しやすい変更に留めること。

#### What changed

- bimanual leader の関節入力と keyboard の episode event を、1 つの HIL-SERL 用 teleop として扱えるようにした。
- human intervention 時に、joint-space の bimanual action をそのまま policy action として流せるようにした。
- 既存の RL dataset に合わせて、最初の学習系は `12-D joint action` 前提で統一した。
- 実機 actor/learner を起動するための bimanual 用 train config を追加した。

#### Why this matters

- これまでは、HIL-SERL の実機入力系が実質的に single-teleop / single-arm 前提だった。
- 今回の変更で、bimanual 実機 RL を始めるための最低限の介入経路が揃った。
- offline dataset と online interaction の action 表現を揃えたため、初期立ち上げ時の action mismatch を避けられる。

#### Design choice

- 既存の `so_leader` や `bi_so_leader` を直接拡張するのではなく、新しい複合 teleop を追加する方式を採用した。
- 理由は、既存コードへの侵入を小さくし、将来 upstream 更新時の conflict を減らすため。
- また、今回は end-effector delta 制御への全面移行は行わず、既存 RL dataset と整合する joint-space action を優先した。

#### Team impact

- bimanual HIL-SERL を試したいメンバーは、新しい train config をベースに実機ポートと camera index を埋めれば起動できる。
- 既存の single-arm や既存 teleop の挙動を置き換える変更ではないため、既存ワークフローへの影響は限定的。
- HIL-SERL の intervention 経路は、今後 bimanual EE 制御へ拡張するための中間段階として整理された。

#### Out of scope in this change

- bimanual end-effector delta action への移行
- reward classifier を使った success 自動判定
- crop 前提の画像最適化や学習性能の追い込み

#### Current assumption

- まずは `bi_so_leader + keyboard` による human intervention が安定して通ることを優先する。
- 学習は、すでに整備済みの RL dataset と同じ joint-space action で開始する。
- no-crop / 高解像度の運用は許容するが、今回の主目的は first working pipeline の確立である。
