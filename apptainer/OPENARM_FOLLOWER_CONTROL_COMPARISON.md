# OpenArm follower制御設定の比較

この文書は、次の4つの動作におけるOpenArm followerの制御設定を比較したものです。

1. `dora-openarm-data-collection`でのbilateral teleop
2. bilateral teleop起動・終了時の姿勢移動CSV再生
3. LeRobotでのpolicyデプロイ（推論中）
4. LeRobot起動・終了時の姿勢移動CSV再生

値は現在の作業ツリーにおける既定値です。配列と表の関節順序は
`J1, J2, J3, J4, J5, J6, J7, gripper`です。Dora/native側の位置はrad、
LeRobotの公開actionと関節制限はmotor-zero基準のdegreeで扱われます。

## 概要

| 動作 | PDゲイン | 重力補償 | 摩擦補償 | 関節角度の扱い |
| --- | --- | --- | --- | --- |
| Dora bilateral teleop | native follower用PD | あり | あり | 通常テレオペ中は後述のCSV用関節制限でclipしない。送信前にDamiao motor固有のMIT制御範囲を検査し、違反時はモータをdisableする |
| Dora 姿勢移動CSV再生 | 原則としてbilateral teleopと同じPD | 原則としてあり | 原則としてあり | 再生開始前にnative側の関節角度・速度制限でCSVを検証する。再生中のtracking error上限は`0.35 rad` |
| LeRobot policyデプロイ | bilateral followerと同じPD | あり | あり | 左右別の安全制限内へ各actionをclipする。`max_relative_target`は既定で無効 |
| LeRobot 姿勢移動CSV再生 | bilateral followerと同じPD | あり | あり | CSVを左右別の制限で事前検証し、送信時にも同じ範囲へclipする。clipまたはtracking errorを検出すると再生を中止する。起動時エラーは両followerをdisableし、終了復帰時エラーは既定で現在姿勢を保持してCAN切断時もトルクを維持する |

LeRobotではpolicy推論とCSV再生のPDゲインを独立して設定できますが、既定値はどちらも
native bilateral followerを正として同じ値に揃えています。MIT制御コマンドの速度目標は
`0`、トルクフィードフォワードはnativeと同じ重力補償と摩擦補償の和です。

## PDゲイン

各セルは`Kp / Kd`です。

| 動作 | J1 | J2 | J3 | J4 | J5 | J6 | J7 | gripper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dora bilateral teleop | `240 / 3` | `240 / 3` | `240 / 3` | `240 / 3` | `24 / 0.2` | `31 / 0.2` | `25 / 0.2` | `16 / 0.2` |
| Dora 姿勢移動CSV再生 | `240 / 3` | `240 / 3` | `240 / 3` | `240 / 3` | `24 / 0.2` | `31 / 0.2` | `25 / 0.2` | `16 / 0.2` |
| LeRobot policyデプロイ | `240 / 3` | `240 / 3` | `240 / 3` | `240 / 3` | `24 / 0.2` | `31 / 0.2` | `25 / 0.2` | `16 / 0.2` |
| LeRobot 姿勢移動CSV再生 | `240 / 3` | `240 / 3` | `240 / 3` | `240 / 3` | `24 / 0.2` | `31 / 0.2` | `25 / 0.2` | `16 / 0.2` |

Doraの既定`J7_TUNING_PROFILE=validated`はfollower J7の`Kp=25`を変更しません。
LeRobotでは次のオプションがそれぞれ対応します。

- policy推論: `robot.{left,right}_arm_config.position_kp`、`position_kd`
- CSV再生: `robot.{left,right}_arm_config.trajectory_position_kp`、`trajectory_position_kd`

### Dora起動時のゼロ姿勢移動の例外

Dora/nativeの起動処理では、CSVを順再生する前にleaderとfollowerを現在姿勢から
設定済みホーム姿勢へ移動します。ホーム姿勢の既定値はmotor zeroです。この区間の
followerは通常のbilateral PDではなく、次の低いゲインを使用します。

| 区間 | J1 | J2 | J3 | J4 | J5 | J6 | J7 | gripper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 現在姿勢からホーム姿勢 | `50 / 1.2` | `50 / 1.2` | `50 / 1.2` | `50 / 1.2` | `10 / 0.3` | `10 / 0.2` | `10 / 0.3` | `10 / 0.5` |

この区間のトルクフィードフォワードは`0`なので、重力補償と摩擦補償は適用されません。
ホーム姿勢に到達した後のCSV順再生と、終了時のタスク姿勢へのblend・CSV逆再生には
通常のDora follower用PD、重力補償、摩擦補償が適用されます。

## 重力補償と摩擦補償

| 動作 | armのトルクフィードフォワード | gripperのトルクフィードフォワード | Coriolis補償 |
| --- | --- | --- | --- |
| Dora bilateral teleop | `gravity + friction` | `friction` | なし（計算するが指令には加算しない） |
| Dora 姿勢移動CSV再生 | `gravity + friction` | `friction` | なし（現在姿勢からホーム姿勢までの例外区間は全項目`0`） |
| LeRobot policyデプロイ | `gravity + friction` | `friction` | なし |
| LeRobot 姿勢移動CSV再生 | `gravity + friction` | `friction` | なし |

Dora/nativeの摩擦モデルは、関節速度を`dq`として次式です。

```text
friction = Fc * tanh(0.1 * k * dq) + Fv * dq + Fo
```

通常起動時の`J7_TUNING_PROFILE=validated`では、左右followerのJ7に次のscaleが
適用されます。J1〜J6のscaleは`1.0`で、gripperは重力補償を持たず、設定ファイルの
摩擦パラメータをそのまま使用します。

| follower J7項目 | follower.yaml値 | scale | 実効値 |
| --- | ---: | ---: | ---: |
| 重力 | dynamicsによる計算値 | `0.95` | `0.95 * gravity` |
| `Fc` | `0.172 Nm` | `0.25` | `0.043 Nm` |
| `k` | `7.888` | `1.0` | `7.888` |
| `Fv` | `0.084 Nm s/rad` | `0.25` | `0.021 Nm s/rad` |
| `Fo` | `-0.059 Nm` | `0.15` | `-0.00885 Nm` |

`J7_TUNING_PROFILE=official`を指定した場合、これらのJ7 scale overrideは適用されず、
`follower.yaml`の値が使用されます。

LeRobotの既定値は、Dora通常起動の`J7_TUNING_PROFILE=validated`を含む上記の値です。
重力項はwrapperがnative `openarm_teleop.sif`内のOpenArm Description 1.0.4から毎回生成する
v10 bimanual URDFを用いて計算します。摩擦項は同じ式とrad/s単位を使用します。したがって、
policy推論とCSV再生のどちらもnative followerと同じ数値・モデルの補償をMITトルク指令へ
加算します。

## 関節角度制限

### Dora/nativeのCSV検証範囲

左右のCSVに同じ範囲を使用します。degreeへ換算した値を示します。

| 関節 | 最小 | 最大 |
| --- | ---: | ---: |
| J1 | `-120 deg` | `120 deg` |
| J2 | `-90 deg` | `180 deg` |
| J3 | `-90 deg` | `90 deg` |
| J4 | `0 deg` | `180 deg` |
| J5 | `-90 deg` | `90 deg` |
| J6 | `-90 deg` | `90 deg` |
| J7 | `-90 deg` | `90 deg` |
| gripper | `-180 deg` | `180 deg` |

CSVのサンプル間速度も全関節で`8 rad/s`以下か検証します。ただし、この角度配列は
通常のbilateral teleop中にfollower目標をclipするためのものではありません。通常制御では
各MIT指令の`Kp`、`Kd`、位置、速度、トルクを検査し、位置については各Damiao motor typeの
`-pMax`から`pMax`までを超えた指令を拒否します。

収録後のLeRobot dataset変換では、既定の`LEROBOT_LIMIT_POLICY=warn`により下記の
LeRobotデプロイ制限に対する監査も行います。これは収録中の実時間制御やclipではありません。

### LeRobotの左右別デプロイ範囲

単位はmotor-zero基準のdegreeです。policy actionと姿勢移動CSVの両方に同じ範囲を
使用します。

| 関節 | 左follower最小 | 左follower最大 | 右follower最小 | 右follower最大 |
| --- | ---: | ---: | ---: | ---: |
| J1 | `-75` | `75` | `-75` | `75` |
| J2 | `-90` | `10` | `-10` | `90` |
| J3 | `-90` | `90` | `-90` | `90` |
| J4 | `0` | `140` | `0` | `140` |
| J5 | `-90` | `90` | `-90` | `90` |
| J6 | `-45` | `45` | `-45` | `45` |
| J7 | `-90` | `90` | `-90` | `90` |
| gripper | `-60` | `0` | `-60` | `0` |

policy actionは範囲外の場合に境界値へclipされます。姿勢移動CSVは再生前に全サンプルを
検証するため、範囲外のCSVはモータを動かす前に拒否されます。また、再生時にも通常の
`send_action()`を経由するためclipが適用されますが、要求値と実際の送信値の不一致を
検出すると安全上のエラーとして再生を中止します。

J1はタスク空間上の制約として左右とも`-75`から`75 deg`を維持し、J2も外向き方向は
左右それぞれ`-90`または`90 deg`までに制限します。J3からJ7はOpenArm Description
v1.0.4の機械的範囲まで拡大しています。blend開始時に読み取った関節角がこの範囲を
既定`1 deg`以内だけ越えている場合は境界値へ丸めます。それ以上の逸脱は動作を開始せず
エラーにします。

終了復帰中のエラーでは、自由落下を避けるため、既定で測定中の現在姿勢をtrajectory用
PDで再指令し、後続のCAN切断でもトルクをdisableしません。プロセス終了後もアームは
通電・保持状態となるため、電源断または手動disableの前に必ず両アームを支えてください。
従来どおりエラー時にdisableする必要がある場合は
`robot.hold_position_on_shutdown_error=false`を指定します。起動時の姿勢移動エラーと
policy action送信エラーは引き続き両followerをdisableします。

## 設定・実装の参照先

- Dora launcherとJ7 profile: `dora-openarm-data-collection/apptainer/run_openarm_teleop_udp_record.sh`
- native followerのPD・摩擦パラメータ: `openarm_teleop/config/follower.yaml`
- native補償・ゼロ姿勢移動・tracking error: `openarm_teleop/src/controller/control.cpp`
- native CSV用関節角度・速度制限: `openarm_teleop/src/openarm_constants.hpp`
- native CSV検証: `openarm_teleop/src/startup_trajectory.cpp`
- LeRobotのPD・左右別関節制限: [`../src/lerobot/robots/openarm_follower/config_openarm_follower.py`](../src/lerobot/robots/openarm_follower/config_openarm_follower.py)
- LeRobotの重力・摩擦補償: [`../src/lerobot/robots/openarm_follower/openarm_dynamics.py`](../src/lerobot/robots/openarm_follower/openarm_dynamics.py)
- LeRobot action送信: [`../src/lerobot/robots/openarm_follower/openarm_follower.py`](../src/lerobot/robots/openarm_follower/openarm_follower.py)
- LeRobot CSV検証: [`../src/lerobot/robots/bi_openarm_follower/deployment_trajectory.py`](../src/lerobot/robots/bi_openarm_follower/deployment_trajectory.py)
- LeRobot CSV再生: [`../src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py`](../src/lerobot/robots/bi_openarm_follower/bi_openarm_follower.py)
