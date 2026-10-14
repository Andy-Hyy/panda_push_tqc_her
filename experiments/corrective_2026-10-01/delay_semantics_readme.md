# delay_steps 的实际含义

诊断序列是 A_t = [0.1×(t+1), 0, 0]，t = 0…11。零向量只留给初始缓冲区，所以日志里的 0 不会和 A0 混在一起。记录的是 wrapper 选出、交给 `FrankaEnv.step` 的动作，发生在环境把动作裁剪到 [-1, 1] 之前。t ≥ 10 的指令超出动作盒，只用来把滞后继续显示出来；t = 0…9 的数值都在盒内。

两种实现都跑了：

- `corrective_deque`：本次 2×2 使用的固定延迟。`delay_steps=0` 执行当前动作；`delay_steps=d` 执行 d 步之前的动作，缓冲区一开始用零填满。
- `stage0_DomainRandomizationWrapper`：`utils/domain_rand.py` 里的原始类。`reset` 之后把 `last_params["delay_steps"]` 设成指定值，再按它的缓冲区下标取动作。

## 结论

`delay_steps=0` 在本次纠正实验里表示没有延迟。诊断里，发出的动作和实际执行的动作逐格相同。

`delay_steps=d`（d ≥ 1）表示滞后 d 个决策。前 d 步执行零动作，之后执行的是 t−d 时刻发出的动作。原始 Stage 0 wrapper 在 d = 1、2、3 上和这个定义逐格一致。

Stage 0 的采样是 `np.random.randint(1, 10)`，只产生 1…9，从不产生 0。因此原始联合实验里的 `delay_steps=1` 不是训练条件，而是整整一个决策的滞后。

原始 wrapper 如果被强行设成 `delay_steps=0`，也不是无延迟。它取 `action_buffer[-delay]`，而 Python 里 `[-0]` 就是 `[0]`，也就是长度 10 的缓冲区里最老的那一格。诊断里这一设置前 10 步都是零，第 10 步才执行 A0，相当于 10 步滞后。2×2 不使用这个下标。

一次 `env.step` 是 10 次 `mj_step`，每次 `nstep=25`，XML 的 `timestep=0.002` 秒。一个决策是 0.5 秒。`delay_steps=d` 就是 0.5×d 秒的指令滞后，不是关节伺服里的毫秒级延迟。

P0 和 P1 使用上面的无延迟。P2 和 P3 使用 Stage 0 CSV 里已经抽好的 1…9，按这里核清的 d 步滞后执行。
