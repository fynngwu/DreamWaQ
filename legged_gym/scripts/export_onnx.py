import os
import sys
import argparse
import glob
import subprocess
import shutil
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from rsl_rl.modules import ActorCriticDreamWaQ


REMOTE = "wufy@100.66.202.29:~/projects/dogv2_mujoco"
LOCAL_DEST = os.path.join(os.path.dirname(__file__), "..", "..", "legged_gym", "scripts", "policy.onnx")


def find_latest_model():
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    all_models = []
    for exp_dir in Path(logs_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if not run_dir.is_dir():
                continue
            models = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=os.path.getmtime, reverse=True)
            if models:
                all_models.append((models[0], run_dir))

    if not all_models:
        raise FileNotFoundError(f"No model_*.pt found under {logs_dir}")

    all_models.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    return all_models[0]


def build_model_from_checkpoint(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    vae_prefix = "vae."
    actor_prefix = "actor."

    vae_keys = {k[len(vae_prefix):]: v for k, v in state_dict.items() if k.startswith(vae_prefix)}
    actor_keys = {k[len(actor_prefix):]: v for k, v in state_dict.items() if k.startswith(actor_prefix)}

    cenet_in_dim = vae_keys["encoder.0.weight"].shape[1]
    num_latent_dims = vae_keys["encode_mean_latent.weight"].shape[0]
    num_est_dims = vae_keys["encode_mean_vel.weight"].shape[0]
    num_obs = vae_keys["decoder.4.weight"].shape[0]

    actor_input_dim = actor_keys["0.weight"].shape[1]
    num_actions = actor_keys["6.weight"].shape[0]
    actor_hidden_dims = [
        actor_keys["0.weight"].shape[0],
        actor_keys["2.weight"].shape[0],
        actor_keys["4.weight"].shape[0],
    ]

    critic_hidden_dims = [
        state_dict["critic.0.weight"].shape[0],
        state_dict["critic.2.weight"].shape[0],
        state_dict["critic.4.weight"].shape[0],
    ]
    num_critic_obs = state_dict["critic.0.weight"].shape[1]

    model = ActorCriticDreamWaQ(
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        cenet_in_dim=cenet_in_dim,
        num_latent_dims=num_latent_dims,
        num_explicit_dims=num_est_dims,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  num_obs={num_obs}, cenet_in_dim={cenet_in_dim}, num_actions={num_actions}")
    print(f"  num_latent_dims={num_latent_dims}, num_est_dims={num_est_dims}")
    print(f"  actor_hidden_dims={actor_hidden_dims}")

    return model.vae, model.actor, num_obs, cenet_in_dim, num_actions


def main():
    parser = argparse.ArgumentParser(description="Export DreamWaQ policy to ONNX")
    parser.add_argument("--output", type=str, default=None, help="Output directory for ONNX and config")
    parser.add_argument("--model", type=str, default=None, help="Path to model.pt (default: auto-find latest across all experiments)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    if args.model is None:
        model_path, latest_run = find_latest_model()
        print(f"Found latest model: {model_path}")
    else:
        model_path = args.model
        latest_run = os.path.dirname(model_path)
        print(f"Using specified model: {model_path}")

    vae, actor, num_obs, cenet_in_dim, num_actions = build_model_from_checkpoint(model_path)

    class ONNXExporter(torch.nn.Module):
        def __init__(self, actor, vae):
            super().__init__()
            self.actor = actor
            self.vae = vae
            self.num_obs = vae.decoder[-1].out_features

        def forward(self, obs_history):
            code, _, _, _ = self.vae.cenet_forward(obs_history)
            obs = obs_history[:, -self.num_obs:]
            actor_input = torch.cat((code, obs), dim=1)
            return self.actor(actor_input)

    exporter = ONNXExporter(actor, vae)

    output_dir = args.output
    if output_dir is None:
        output_dir = Path(model_path).parent
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "policy.onnx")
    dummy = torch.randn(1, cenet_in_dim)

    torch.onnx.export(
        exporter,
        (dummy,),
        onnx_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"ONNX exported to: {onnx_path}")
    print(f"  Input (obs_history):     {cenet_in_dim}")
    print(f"  Output (actions):        {num_actions}")

    info_path = os.path.join(output_dir, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Source model: {model_path}\n")
        f.write(f"num_obs: {num_obs}\n")
        f.write(f"cenet_in_dim (history): {cenet_in_dim}\n")
        f.write(f"num_actions: {num_actions}\n")
    print(f"Model info saved to: {info_path}")

    if os.path.exists(LOCAL_DEST):
        os.remove(LOCAL_DEST)
    shutil.copy2(onnx_path, LOCAL_DEST)
    print(f"Copied to local: {LOCAL_DEST}")

    print(f"\nCopying to remote {REMOTE}...")
    result = subprocess.run(
        ["scp", LOCAL_DEST, REMOTE],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print(f"Copied to remote: {REMOTE}")
    else:
        print(f"Remote copy failed (return code {result.returncode}): {result.stderr.strip()}")

    print("\nUsage on dog_v2_mujoco/rl_sar-ARES:")
    print(f"  Input (obs_history):     {cenet_in_dim} (full history buffer)")
    print(f"  Set num_observations:    {num_obs}")
    print(f"  obs_history size:        {cenet_in_dim}")


if __name__ == "__main__":
    main()
