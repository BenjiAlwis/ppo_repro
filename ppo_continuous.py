"""
ppo_continuous.py
=================
Paper-faithful PPO for continuous control (MuJoCo / Gymnasium), written for the
CS5180 capstone reproduction of Schulman et al. (2017),
"Proximal Policy Optimization Algorithms".

Single script, three objective modes selected with --mode:
    clip    : the clipped surrogate objective  L^CLIP   (the paper's main method)
    noclip  : no clipping, no penalty          L^CPI    (ablation baseline)
    kl      : adaptive KL-penalty objective     L^KLPEN  (ablation baseline)

The data collection, network, GAE, and optimisation loop are IDENTICAL across
modes; only the policy-loss term changes. This is what makes the ablation fair:
any difference in results is attributable to the objective alone.

Logging: always writes a CSV (never fails); TensorBoard if available.

Default hyperparameters follow the paper's MuJoCo Table 3:
    horizon T = 2048, Adam lr = 3e-4, epochs K = 10, minibatch = 64,
    gamma = 0.99, GAE lambda = 0.95, clip epsilon = 0.2, 1M timesteps.

Implementation details that the paper omits but that matter for performance
(documented in the code where they occur):
    - advantage normalisation per minibatch
    - orthogonal weight init with tuned gains
    - value-function loss clipping
    - global gradient-norm clipping
    - learning-rate annealing
    - separate (non-shared) policy and value networks, as the paper specifies
      for the MuJoCo experiments ("we don't share parameters ... c1 is irrelevant")

Usage:
    python ppo_continuous.py --env-id HalfCheetah-v5 --mode clip --seed 1
    python ppo_continuous.py --env-id Hopper-v5 --mode kl --seed 2 --total-timesteps 1000000
"""

import os

# Headless rendering for servers without a display (RunPod).
# MuJoCo's env import initialises an OpenGL context even during training (when
# nothing is actually rendered), so on a headless box the import can crash
# unless we select the EGL backend first. Set before importing gymnasium.
# Override by exporting MUJOCO_GL yourself (e.g. "glfw" on a local desktop).
os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import time
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym


def _default_log_dir():
    """
    Prefer the persistent volume on RunPod (/workspace) so results survive a
    pod restart; otherwise log to ./runs locally. Override with --log-dir.
    """
    if os.path.isdir("/workspace"):
        return "/workspace/runs"
    return "runs"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    env_id: str = "HalfCheetah-v5"
    mode: str = "clip"                 # clip | noclip | kl
    seed: int = 1
    total_timesteps: int = 1_000_000
    torch_deterministic: bool = True
    cuda: bool = True

    # PPO core (paper Table 3)
    num_envs: int = 1                  # vectorised envs; paper's single-actor MuJoCo setup
    num_steps: int = 2048              # horizon T (rollout length per env)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32          # -> minibatch size = (num_envs*num_steps)/num_minibatches = 64
    update_epochs: int = 10            # K
    learning_rate: float = 3e-4
    anneal_lr: bool = True

    # objective-specific
    clip_coef: float = 0.2             # epsilon (used when mode == clip)
    kl_target: float = 0.01            # d_targ for adaptive KL (used when mode == kl)
    kl_beta_init: float = 1.0          # initial KL penalty coefficient

    # shared loss terms
    ent_coef: float = 0.0              # paper uses NO entropy bonus for MuJoCo
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True            # value-function loss clipping (implementation detail)
    norm_adv: bool = True              # advantage normalisation (implementation detail)

    # logging
    log_dir: str = ""                  # "" -> auto (/workspace/runs on RunPod, else runs)
    eval_interval_updates: int = 0     # 0 disables periodic greedy eval

    # derived at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def make_env(env_id, seed, idx, run_name, capture_video=False):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # These wrappers are standard for MuJoCo PPO and part of the "implementation
        # details" that make results reproducible. They are NOT in the paper.
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        env.action_space.seed(seed + idx)
        return env
    return thunk


# --------------------------------------------------------------------------- #
# Networks (separate policy and value, as the paper specifies for MuJoCo)
# --------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal init with tuned gains — a well-known PPO implementation detail.
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Value network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Policy network — outputs the mean of a Gaussian; log-std is a free
        # state-independent parameter (matches the paper's Gaussian policy).
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        # sum over action dimensions -> joint log-prob of the action vector
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x)


# --------------------------------------------------------------------------- #
# CSV logger (never fails — this is the fallback that always gives raw numbers)
# --------------------------------------------------------------------------- #
class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["global_step", "metric", "value"])
        self._f.flush()

    def log(self, global_step, metric, value):
        self._w.writerow([global_step, metric, float(value)])
        self._f.flush()

    def close(self):
        self._f.close()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: Args):
    # Resolve auto log dir ("" -> /workspace/runs on RunPod, else runs)
    if not args.log_dir:
        args.log_dir = _default_log_dir()

    run_name = f"{args.env_id}__{args.mode}__seed{args.seed}__{int(time.time())}"

    # Derived sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[{run_name}] device={device} batch={args.batch_size} "
          f"minibatch={args.minibatch_size} updates={args.num_updates}")

    # Loggers
    csv_path = os.path.join(args.log_dir, run_name, "metrics.csv")
    csv_logger = CSVLogger(csv_path)
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(args.log_dir, run_name, "tb"))
        tb_writer.add_text("hyperparameters",
                           "\n".join([f"{k}={v}" for k, v in asdict(args).items()]))
    except Exception as e:
        print(f"[warn] TensorBoard unavailable ({e}); CSV logging only.")

    # Vectorised envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action space required"
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # KL-penalty coefficient (only used in kl mode; adapted each update)
    kl_beta = args.kl_beta_init

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for update in range(1, args.num_updates + 1):
        # Learning-rate annealing (implementation detail; paper anneals for Atari)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- Rollout: collect num_steps of on-policy data ------------------ #
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # Episode returns come from RecordEpisodeStatistics (raw, un-normalised)
            if "episode" in infos:
                # SyncVectorEnv puts a mask in infos["_episode"]
                mask = infos.get("_episode", None)
                ep_returns = infos["episode"]["r"]
                ep_lengths = infos["episode"]["l"]
                for i in range(args.num_envs):
                    if mask is None or mask[i]:
                        r_i = float(ep_returns[i]); l_i = float(ep_lengths[i])
                        csv_logger.log(global_step, "episodic_return", r_i)
                        csv_logger.log(global_step, "episodic_length", l_i)
                        if tb_writer:
                            tb_writer.add_scalar("charts/episodic_return", r_i, global_step)
                            tb_writer.add_scalar("charts/episodic_length", l_i, global_step)

        # ---- GAE: compute advantages and returns -------------------------- #
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # ---- Flatten the batch -------------------------------------------- #
        b_obs = obs.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1, act_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- Optimise for K epochs over minibatches ----------------------- #
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl_epoch = 0.0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Diagnostic KL (Schulman's low-variance estimator) and clip fraction
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_epoch = approx_kl.item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ===== THE ONLY PART THAT DIFFERS BY MODE ===================== #
                if args.mode == "clip":
                    # L^CLIP: pessimistic min of clipped and unclipped surrogate
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                elif args.mode == "noclip":
                    # L^CPI: plain importance-weighted surrogate, no clip, no penalty
                    pg_loss = -(mb_adv * ratio).mean()
                elif args.mode == "kl":
                    # L^KLPEN: surrogate minus beta * KL(old || new)
                    # approx_kl >= 0 estimates KL(old||new); beta adapted after the update
                    kl_penalty = ((ratio - 1) - logratio).mean()  # same estimator, keeps grad
                    pg_loss = -(mb_adv * ratio).mean() + kl_beta * kl_penalty
                else:
                    raise ValueError(f"unknown mode {args.mode}")
                # ============================================================= #

                # Value loss (optionally clipped — implementation detail)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- Adaptive KL coefficient update (kl mode only) ---------------- #
        if args.mode == "kl":
            # Paper's rule: if measured KL too low, halve beta; if too high, double it.
            d = max(approx_kl_epoch, 0.0)
            if d < args.kl_target / 1.5:
                kl_beta = max(kl_beta / 2.0, 1e-4)
            elif d > args.kl_target * 1.5:
                kl_beta = min(kl_beta * 2.0, 1e4)

        # ---- Diagnostics --------------------------------------------------- #
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))

        csv_logger.log(global_step, "value_loss", v_loss.item())
        csv_logger.log(global_step, "policy_loss", pg_loss.item())
        csv_logger.log(global_step, "approx_kl", approx_kl_epoch)
        csv_logger.log(global_step, "clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0)
        csv_logger.log(global_step, "explained_variance", float(explained_var))
        csv_logger.log(global_step, "sps", sps)
        if args.mode == "kl":
            csv_logger.log(global_step, "kl_beta", kl_beta)
        if tb_writer:
            tb_writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            tb_writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            tb_writer.add_scalar("losses/approx_kl", approx_kl_epoch, global_step)
            tb_writer.add_scalar("losses/clipfrac",
                                 float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
            tb_writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
            tb_writer.add_scalar("charts/SPS", sps, global_step)
            if args.mode == "kl":
                tb_writer.add_scalar("charts/kl_beta", kl_beta, global_step)

        if update % 10 == 0 or update == 1:
            print(f"[{args.env_id}|{args.mode}|s{args.seed}] "
                  f"update {update}/{args.num_updates} step {global_step} "
                  f"vloss {v_loss.item():.3f} kl {approx_kl_epoch:.4f} "
                  f"expvar {explained_var:.3f} sps {sps}")

    envs.close()
    csv_logger.close()
    if tb_writer:
        tb_writer.close()
    print(f"[{run_name}] done in {time.time() - start_time:.0f}s")
    return csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Paper-faithful PPO (continuous control) with ablation modes.")
    d = Args()
    p.add_argument("--env-id", type=str, default=d.env_id)
    p.add_argument("--mode", type=str, default=d.mode, choices=["clip", "noclip", "kl"])
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--total-timesteps", type=int, default=d.total_timesteps)
    p.add_argument("--num-steps", type=int, default=d.num_steps)
    p.add_argument("--num-envs", type=int, default=d.num_envs)
    p.add_argument("--num-minibatches", type=int, default=d.num_minibatches)
    p.add_argument("--update-epochs", type=int, default=d.update_epochs)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    p.add_argument("--clip-coef", type=float, default=d.clip_coef)
    p.add_argument("--kl-target", type=float, default=d.kl_target)
    p.add_argument("--ent-coef", type=float, default=d.ent_coef)
    p.add_argument("--vf-coef", type=float, default=d.vf_coef)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--no-norm-adv", action="store_true")
    p.add_argument("--no-clip-vloss", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-dir", type=str, default=d.log_dir,
                   help="output dir; default auto: /workspace/runs on RunPod, else runs")
    a = p.parse_args()

    args = Args(
        env_id=a.env_id, mode=a.mode, seed=a.seed, total_timesteps=a.total_timesteps,
        num_steps=a.num_steps, num_envs=a.num_envs, num_minibatches=a.num_minibatches,
        update_epochs=a.update_epochs, learning_rate=a.learning_rate, gamma=a.gamma,
        gae_lambda=a.gae_lambda, clip_coef=a.clip_coef, kl_target=a.kl_target,
        ent_coef=a.ent_coef, vf_coef=a.vf_coef, log_dir=a.log_dir,
        anneal_lr=not a.no_anneal_lr, norm_adv=not a.no_norm_adv,
        clip_vloss=not a.no_clip_vloss, cuda=not a.cpu,
    )
    return args


if __name__ == "__main__":
    train(parse_args())
