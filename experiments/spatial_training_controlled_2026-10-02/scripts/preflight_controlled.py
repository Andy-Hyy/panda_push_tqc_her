"""Implementation gate for the controlled SpatialDR and Fixed Large retraining.

Formal training is locked only after the range checks and both technical smokes pass.
A smoke success rate is recorded and then ignored.
"""

from __future__ import annotations

import csv
import hashlib
import os
import traceback

import numpy as np

import controlled_common as common
import eval_controlled
import train_controlled

RANGE_FIELDS = (
    "kind",
    "index",
    "reset_seed",
    "requested_L",
    "actual_L",
    "goal_low_x",
    "goal_low_y",
    "goal_low_z",
    "goal_high_x",
    "goal_high_y",
    "goal_high_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "desired_x",
    "desired_y",
    "desired_z",
    "object_x",
    "object_y",
    "object_z",
    "goal_in_box",
    "object_in_box",
    "attribute_matches",
)


def add_problem(problems: list[str], label: str, fn):
    try:
        return fn()
    except SystemExit as exc:
        problems.append(f"{label}: {exc}")
    except Exception as exc:
        problems.append(f"{label}: {type(exc).__name__}: {exc}")
    return None


def range_row(kind: str, index: int, reset_seed: int, requested: float, base, obs) -> dict:
    pose = common.pose_after_reset(base, obs)
    low = pose["goal_low"]
    high = pose["goal_high"]
    goal = pose["goal"]
    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
    obj = pose["object"]
    return {
        "kind": kind,
        "index": index,
        "reset_seed": reset_seed,
        "requested_L": float(requested),
        "actual_L": float(base.goal_xy_range),
        "goal_low_x": float(low[0]),
        "goal_low_y": float(low[1]),
        "goal_low_z": float(low[2]),
        "goal_high_x": float(high[0]),
        "goal_high_y": float(high[1]),
        "goal_high_z": float(high[2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
        "desired_x": float(desired[0]),
        "desired_y": float(desired[1]),
        "desired_z": float(desired[2]),
        "object_x": float(obj[0]),
        "object_y": float(obj[1]),
        "object_z": float(obj[2]),
        "goal_in_box": int(common.in_box(goal, low, high)),
        "object_in_box": int(common.in_box(obj, pose["obj_low"], pose["obj_high"])),
        "attribute_matches": int(abs(float(base.goal_xy_range) - float(requested)) == 0.0),
    }


def write_rows(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def spatial_reference() -> dict:
    path = common.PANDA_FINAL / "data_and_analysis" / "spatial_clean_2026-10-02" / "spatial_seed0.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["episode_id"]) == 0 and int(row["reset_seed"]) == 6_030_000:
                return row
    raise SystemExit("spatial_seed0 episode 0 at reset_seed 6030000 was not found")


def pose_of(row: dict) -> np.ndarray:
    names = (
        "initial_object_x",
        "initial_object_y",
        "initial_object_z",
        "target_x",
        "target_y",
        "target_z",
    )
    return np.array([float(row[name]) for name in names], dtype=np.float64)


def collect_poses(side: float, seeds: list[int]) -> dict[int, np.ndarray]:
    env, base, _pristine, _ids = common.make_push_env(0.30)
    found = {}
    try:
        for seed in seeds:
            common.assert_side(base, side)
            obs, _info = env.reset(seed=int(seed))
            pose = common.pose_after_reset(base, obs)
            found[int(seed)] = np.concatenate([pose["object"], pose["goal"]])
    finally:
        env.close()
    return found


def run_logic() -> tuple[list[str], list[dict], dict]:
    problems: list[str] = []
    rows: list[dict] = []
    notes: dict = {}
    common.ensure_mjb()
    if common.final_reset_seed(0.30, 0) != 6_030_000:
        problems.append(f"final seed formula gave {common.final_reset_seed(0.30, 0)} for L=0.30 episode 0")
    validation = {common.validation_reset_seed(side, episode) for side in common.SIDES for episode in range(100)}
    final = {common.final_reset_seed(side, episode) for side in common.SIDES for episode in range(600)}
    if not validation.isdisjoint(final):
        problems.append("validation seeds overlap final-test seeds")
    notes["validation_seeds"] = len(validation)
    notes["final_seeds"] = len(final)

    def forced():
        env, base, _pristine, _ids = common.make_push_env(0.30)
        try:
            for side in (0.30, 0.55, 0.80):
                for index, seed in enumerate((101, 102, 103)):
                    common.assert_side(base, side)
                    obs, _info = env.reset(seed=seed)
                    row = range_row("forced", index, seed, side, base, obs)
                    rows.append(row)
                    if row["goal_in_box"] != 1 or row["object_in_box"] != 1 or row["attribute_matches"] != 1:
                        raise SystemExit(f"forced L={side} seed {seed} failed the box readback")
                    span = np.array([row["goal_high_x"] - row["goal_low_x"], row["goal_high_y"] - row["goal_low_y"]])
                    if not np.allclose(span, [side, side], rtol=0.0, atol=1e-8):
                        raise SystemExit(f"forced span {span} for L={side}")
        finally:
            env.close()

    add_problem(problems, "forced range", forced)

    def attribute_only():
        env, base, _pristine, _ids = common.make_push_env(0.30)
        try:
            outside = 0
            for index in range(20):
                base.goal_xy_range = 0.80
                obs, _info = env.reset(seed=5000 + index)
                row = range_row("attribute_only", index, 5000 + index, 0.80, base, obs)
                rows.append(row)
                bounds = np.array([row["goal_low_x"], row["goal_low_y"], row["goal_high_x"], row["goal_high_y"]])
                if not np.allclose(bounds, [0.45, -0.15, 0.75, 0.15], rtol=0.0, atol=1e-8):
                    raise SystemExit(f"attribute-only reset changed the sampling bounds to {bounds}")
                goal = np.array([row["goal_x"], row["goal_y"]])
                if not (0.45 - 1e-9 <= goal[0] <= 0.75 + 1e-9 and -0.15 - 1e-9 <= goal[1] <= 0.15 + 1e-9):
                    outside += 1
            notes["attribute_only_goals_outside_0.30"] = outside
            if outside != 0:
                raise SystemExit("attribute-only goals left the original 0.30 m box")
        finally:
            env.close()

    add_problem(problems, "attribute only", attribute_only)

    def automatic(method: str, count: int):
        log_path = common.OUT / "preflight" / f"wrapper_{method}.csv"
        env, _base, _pristine, _ids = common.make_push_env(0.30)
        env = train_controlled.Float32Wrapper(env)
        env = train_controlled.GoalDistribution(
            env,
            method=method,
            train_seed=0,
            env_index=0,
            log_path=log_path,
        )
        drawn = []
        try:
            for index in range(count):
                seed = 20_000 + index
                obs, _info = env.reset(seed=seed)
                episode = env.open_episode
                row = range_row(method, index, seed, float(episode["L"]), env.unwrapped, obs)
                rows.append(row)
                drawn.append(float(episode["L"]))
                if row["goal_in_box"] != 1 or row["object_in_box"] != 1:
                    raise SystemExit(f"{method} reset {index} left a box")
                if abs(row["actual_L"] - float(episode["L"])) != 0.0:
                    raise SystemExit(f"{method} attribute does not match the drawn L")
        finally:
            env.close()
        return drawn

    dr = add_problem(problems, "spatial_dr resets", lambda: automatic("spatial_dr", 100))
    fixed = add_problem(problems, "fixed_large resets", lambda: automatic("fixed_large", 30))
    if dr is not None:
        replay_rng = np.random.Generator(np.random.PCG64(common.l_seed(0, 0)))
        replay = [float(replay_rng.uniform(common.DR_LOW, common.DR_HIGH)) for _ in range(len(dr))]
        if replay != dr:
            problems.append("DR side sequence does not replay from the dedicated generator")
        if min(dr) >= 0.5 or max(dr) <= 0.6 or len(set(dr)) < 2:
            problems.append(f"DR sides did not vary across the box: min {min(dr)} max {max(dr)}")
        if any(not (common.DR_LOW <= value < common.DR_HIGH) for value in dr):
            problems.append("a DR side fell outside [0.30, 0.80)")
        notes["dr_min"] = min(dr)
        notes["dr_max"] = max(dr)
        notes["dr_count"] = len(dr)
    if fixed is not None:
        if any(value != common.FIXED_SIDE for value in fixed):
            problems.append("a Fixed Large episode did not record L=0.80")
        outside = [
            row
            for row in rows
            if row["kind"] == "fixed_large"
            and (row["goal_x"] < 0.45 - 1e-9 or row["goal_x"] > 0.75 + 1e-9 or abs(row["goal_y"]) > 0.15 + 1e-9)
        ]
        notes["fixed_goals_outside_0.30"] = len(outside)
        if not outside:
            problems.append("every Fixed Large goal stayed inside the 0.30 m box")

    def replay_poses():
        seeds = [11, 12, 13, 14]
        forward = collect_poses(0.55, seeds)
        backward = collect_poses(0.55, list(reversed(seeds)))
        again = collect_poses(0.55, seeds)
        deltas = []
        for seed in seeds:
            deltas.append(float(np.max(np.abs(forward[seed] - backward[seed]))))
            deltas.append(float(np.max(np.abs(forward[seed] - again[seed]))))
        notes["reverse_replay_maxabs"] = max(deltas)
        if max(deltas) > 1e-8:
            raise SystemExit(f"reverse replay maxabs {max(deltas)}")
        env, base, _pristine, _ids = common.make_push_env(0.30)
        try:
            common.assert_side(base, 0.55)
            first, _info = env.reset(seed=11)
            first_pose = np.concatenate(
                [common.pose_after_reset(base, first)["object"], common.pose_after_reset(base, first)["goal"]]
            )
            common.assert_side(base, 0.80)
            env.reset(seed=12)
            common.assert_side(base, 0.55)
            second, _info = env.reset(seed=11)
            second_pose = np.concatenate(
                [common.pose_after_reset(base, second)["object"], common.pose_after_reset(base, second)["goal"]]
            )
        finally:
            env.close()
        repeat_delta = float(np.max(np.abs(first_pose - second_pose)))
        notes["same_env_repeat_maxabs"] = repeat_delta
        if repeat_delta > 1e-8:
            raise SystemExit(f"same-env repeat maxabs {repeat_delta}")

    add_problem(problems, "replay", replay_poses)

    def reward_and_delay():
        common.import_helpers()
        import phaseB_diagnose as phase_b

        env, base, _pristine, _ids = common.make_push_env(0.30)
        env = train_controlled.Float32Wrapper(env)
        env = train_controlled.GoalDistribution(
            env,
            method="fixed_large",
            train_seed=0,
            env_index=1,
            log_path=common.OUT / "preflight" / "wrapper_reward.csv",
        )
        try:
            achieved = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
            desired = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.051, 0.0, 0.0]], dtype=np.float64)
            reward_fn = env.get_wrapper_attr("compute_reward")
            got = np.asarray(reward_fn(achieved, desired, [{}, {}, {}]), dtype=np.float64)
            expected = -(np.linalg.norm(achieved - desired, axis=-1) > 0.05).astype(np.float64)
            if got.shape != expected.shape or not np.array_equal(got, expected):
                raise SystemExit(f"HER compute_reward {got} != environment rule {expected}")
            direct = np.asarray(base.compute_reward(achieved, desired, {}), dtype=np.float64)
            if not np.array_equal(direct, expected):
                raise SystemExit("unwrapped compute_reward disagrees with the sparse rule")
            obs, _info = env.reset(seed=77)
            action = np.array([0.2, -0.1, 0.05], dtype=np.float32)
            delay = phase_b.Delay(0, 3)
            executed = np.asarray(delay.apply(action), dtype=np.float32)
            if not np.array_equal(executed, action):
                raise SystemExit("Delay(0) changed the action")
            _obs, reward, _terminated, _truncated, info = env.step(executed)
            if not np.isfinite(float(reward)):
                raise SystemExit("reward was not finite")
            if "is_success" not in info:
                raise SystemExit("step info has no is_success")
        finally:
            env.close()

    add_problem(problems, "reward and delay", reward_and_delay)

    def compatibility():
        reference = spatial_reference()
        expected = pose_of(reference)
        env, base, pristine, ids = common.make_push_env(0.30)
        try:
            common.import_helpers()
            import phaseB_diagnose as phase_b
            import phaseE_formal as phase_e

            common.assert_side(base, 0.30)
            obs, _info = env.reset(seed=6_030_000)
            pose = common.pose_after_reset(base, obs)
            actual = np.concatenate([pose["object"], pose["goal"]])
            delta = float(np.max(np.abs(actual - expected)))
            notes["spatial_pose_maxabs"] = delta
            notes["spatial_initial_distance"] = float(pose["distance"])
            if delta > 1e-8:
                raise SystemExit(
                    f"spatial scene maxabs {delta}; expected {expected.tolist()} actual {actual.tolist()}"
                )
            if abs(float(pose["distance"]) - float(reference["initial_distance"])) > 1e-8:
                raise SystemExit("initial distance disagrees with spatial_seed0")
            measured = phase_e.measure(
                base,
                {
                    "state_after_reset_full": phase_b.capture_state(
                        base, {"obj_body": ids["body"], "obj_geom": ids["geom"]}
                    ),
                    "achieved_after_reset": pose["object"],
                    "desired_after_reset": pose["goal"],
                },
                pristine,
                ids,
                common.PHYSICS,
            )
            notes["nominal_mass"] = float(measured["mass_read"])
            notes["nominal_friction"] = float(measured["friction_read"])
            notes["nominal_damping_scale"] = float(measured["damping_scale_read"])
            notes["nominal_gain_scale"] = float(measured["gain_scale_read"])
        finally:
            env.close()

    add_problem(problems, "spatial compatibility", compatibility)
    notes["problem_count"] = len(problems)
    return problems, rows, notes


def file_hashes() -> dict[str, str]:
    common.ensure_mjb()
    return {name: common.sha256_file(path) for name, path in common.implementation_paths().items()}


def lock_protocol(observed: dict) -> dict:
    files = file_hashes()
    body = {
        "protocol_id": common.PROTOCOL_ID,
        "methods": list(common.METHODS),
        "train_seeds": list(common.TRAIN_SEEDS),
        "n_envs": common.N_ENVS,
        "requested_timesteps": common.TOTAL_TIMESTEPS,
        "actual_stop_timesteps": common.CHECKPOINT_STEPS[-1],
        "checkpoint_steps": list(common.CHECKPOINT_STEPS),
        "sampling": common.SAMPLING,
        "physics": common.PHYSICS,
        "horizon": common.HORIZON,
        "delay_steps": 0,
        "success": "distance strictly < 0.05 after at least one env.step",
        "reward": common.TRAIN_CONFIG["reward_rule"],
        "train_config": common.TRAIN_CONFIG,
        "observed_defaults": observed,
        "validation_reset_seed": "7000000 + round(L*100)*1000 + episode_id, episode_id 0..99",
        "final_reset_seed": "6000000 + round(L*100)*1000 + episode_id, episode_id 0..599",
        "validation_episodes_per_cell": 100,
        "final_episodes_per_cell": 600,
        "primary_final_comparison": "L=0.80 mean of the three training seeds and the paired same-seed difference",
        "secondary_comparison": "equal-weight mean over L=0.30..0.80 and the eight-length curve",
        "error_bar": "sample SD of the three policy rates, ddof=1, percentage points; not a confidence interval",
        "near_start": "initial_distance < 0.05; the sensitivity report drops those episodes and states the denominator",
        "representative_rule": eval_controlled.REPRESENTATIVE_RULE,
        "failure_rule": eval_controlled.FAILURE_RULE,
        "resume": "Checkpoints exclude the HER replay buffer. An interrupted formal run is renamed and restarted.",
        "smoke": "20,004-step runs are implementation checks and are not formal results",
        "historical_reference": "The three standard policies are not rerun and are not mixed into these six models",
        "imports": {name: str(path) for name, path in common.dependency_paths().items()},
        "mjb_source": str(common.SPATIAL_MJB),
        "files": files,
        "packages": common.package_versions(),
    }
    body["protocol_sha256"] = hashlib.sha256(common.canonical(body).encode()).hexdigest()
    text = common.canonical(body)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(common.PROTOCOL_PATH, flags, 0o644)
    except FileExistsError:
        existing = common.PROTOCOL_PATH.read_text()
        if existing != text:
            raise SystemExit("protocol file already exists and differs")
        return body
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)
    return body


def write_report(path, logic_notes: dict, problems: list[str], smoke: dict, protocol: dict | None) -> None:
    passed = not problems and protocol is not None
    lines = [
        "# 受控重训预检",
        "",
        "这轮要训练的是六个新模型。SpatialDR 和 Fixed Large 共用同一套清洁初始化和 TQC+HER，只在每局目标区域边长的取法上不同。历史训练脚本没有被改，也没有被拿来续训。",
        "",
        f"实现门槛{'已通过' if passed else '未通过'}。正式训练{'可以按冻结协议开始' if passed else '没有启动'}。",
        "",
        "## 范围",
        "",
        "强制边长 0.30、0.55 和 0.80 时，采样前写入的是 `goal_range_low/high`，不是只改 `goal_xy_range`。返回观测里的 `desired_goal` 与环境里的目标一致，物体区域保持 0.30 m。SpatialDR 的 100 次自动 reset 使用独立播种的 numpy 生成器，边长落在半开区间 [0.30, 0.80)，同一生成器可以原样重放。Fixed Large 的记录边长都是 0.80，而且目标会出现在 0.30 m 盒子之外。",
        "",
        f"只改属性、不改采样边界的对照仍然复现了旧行为：边界停在 0.30 m，20 次目标都没有离开这个盒子（跑出盒子的次数是 {logic_notes.get('attribute_only_goals_outside_0.30', '未完成')}）。所以这轮的 setter 必须改真正参与采样的数组。",
        "",
        f"DR 抽到的边长范围是 {logic_notes.get('dr_min', '未完成')} 到 {logic_notes.get('dr_max', '未完成')}，共 {logic_notes.get('dr_count', 0)} 次。Fixed Large 落在 0.30 m 盒子外的目标有 {logic_notes.get('fixed_goals_outside_0.30', '未完成')} 个。逆序重放的最大坐标差是 {logic_notes.get('reverse_replay_maxabs', '未完成')}，同一环境重复同一种子的最大差是 {logic_notes.get('same_env_repeat_maxabs', '未完成')}。",
        "",
        "## 与已有空间场景",
        "",
        f"L=0.30、reset_seed=6030000 的初始物体和目标，与 `spatial_seed0.csv` 第一局的最大绝对差是 {logic_notes.get('spatial_pose_maxabs', '未完成')}。名义物理读回是质量 {logic_notes.get('nominal_mass', '未完成')}、摩擦 {logic_notes.get('nominal_friction', '未完成')}、阻尼倍率 {logic_notes.get('nominal_damping_scale', '未完成')}、增益倍率 {logic_notes.get('nominal_gain_scale', '未完成')}。延迟 0 原样执行动作。HER 取到的 `compute_reward` 与环境的稀疏规则一致：距离大于 0.05 记 -1，否则记 0。成功仍是严格小于 0.05，所以距离刚好等于 0.05 时奖励为 0、成功为假。",
        "",
        "## 技术试跑",
        "",
    ]
    if smoke:
        lines.append(
            f"两组各种子 0 的技术试跑都停在 {smoke.get('steps')} 步。解析出来的网络、分位数和优化器默认值一致。试跑检查点可以载入，并在独立探针种子上各完成了两局，步数至少为 1，距离为有限值。探针成功与否不参与是否开训的判断，两组探针成功数是 {smoke.get('probe_success')}。"
        )
    else:
        lines.append("技术试跑没有完成，正式协议也没有锁定。")
    if protocol is not None:
        lines.extend(
            [
                "",
                f"冻结协议是 `{protocol['protocol_id']}`，SHA-256 为 `{protocol['protocol_sha256']}`。预定预算的实际停止步数是 {protocol['actual_stop_timesteps']}。检查点不含 HER 回放，中断后的正式运行会整段重开，不把残缺续跑当成同一次训练。",
            ]
        )
    if problems:
        lines.extend(["", "## 未通过的项目", ""])
        lines.extend(f"- {item}" for item in problems)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if common.PROTOCOL_PATH.exists():
        protocol = common.assert_protocol_code()
        print(common.canonical({"protocol": "locked", "sha256": protocol["protocol_sha256"]}))
        return
    problems, rows, notes = run_logic()
    range_path = common.OUT / "range_readback.csv"
    write_rows(range_path, rows)
    smoke_info = {}
    protocol = None
    if not problems:
        observed = None
        for method in common.METHODS:
            try:
                meta = train_controlled.run_training(method, 0, smoke=True)
            except Exception:
                problems.append(f"smoke {method}:\n{traceback.format_exc()}")
                continue
            if int(meta["num_timesteps"]) != common.SMOKE_EXPECTED_STEPS:
                problems.append(f"smoke {method} stopped at {meta['num_timesteps']}")
                continue
            if observed is None:
                observed = meta["resolved"]
            elif meta["resolved"] != observed:
                problems.append("the two arms resolved different hyperparameters")
            try:
                probed = eval_controlled.probe_checkpoint(common.OUT / "smoke" / f"{method}_seed0" / "final_model.zip", 2)
            except Exception:
                problems.append(f"probe {method}:\n{traceback.format_exc()}")
                continue
            for row in probed:
                common.append_csv(common.OUT / "smoke" / "probe_rows.csv", {field: row[field] for field in eval_controlled.EVAL_FIELDS})
            smoke_info.setdefault("probe_success", [])
            smoke_info["probe_success"].extend(int(row["success"]) for row in probed)
            smoke_info["steps"] = int(meta["num_timesteps"])
        if observed is not None and not problems:
            protocol = lock_protocol(observed)
    gate = {
        "passed": not problems and protocol is not None,
        "problems": problems,
        "logic": notes,
        "smoke": smoke_info,
        "protocol_sha256": None if protocol is None else protocol["protocol_sha256"],
        "range_rows": len(rows),
    }
    (common.OUT / "gate_result.json").write_text(common.canonical(gate))
    write_report(common.OUT / "preflight_report.md", notes, problems, smoke_info, protocol)
    if problems:
        raise SystemExit("preflight failed: " + " | ".join(problems[:4]))
    print(common.canonical({"passed": True, "protocol_sha256": protocol["protocol_sha256"]}))


if __name__ == "__main__":
    main()
