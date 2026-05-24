import os
import sys
import argparse
import glob
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch


def find_latest_model(experiment_name):
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs", experiment_name)
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    run_dirs = sorted(Path(log_dir).iterdir(), key=os.path.getmtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {log_dir}")

    latest_run = run_dirs[0]
    models = sorted(glob.glob(os.path.join(latest_run, "model_*.pt")), key=os.path.getmtime, reverse=True)
    if not models:
        raise FileNotFoundError(f"No model_*.pt found in {latest_run}")

    return models[0], latest_run


def main():
    parser = argparse.ArgumentParser(description="Export DreamWaQ policy to ONNX (supports any num_obs)")
    parser.add_argument("--task", type=str, default="dog_v2_gait", help="Task name for auto-finding latest model")
    parser.add_argument("--output", type=str, default=None, help="Output directory for ONNX and config")
    parser.add_argument("--model", type=str, default=None, help="Path to model.pt (auto-find latest if not given)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    if args.model is None:
        model_path, latest_run = find_latest_model(args.task)
        print(f"Found latest model: {model_path}")
    else:
        model_path = args.model
        latest_run = os.path.dirname(model_path)
        print(f"Using specified model: {model_path}")

    model = torch.jit.load(model_path, map_location="cpu")

    if hasattr(model, 'vae') and hasattr(model, 'actor'):
        vae = model.vae
        actor = model.actor
    elif hasattr(model, 'actor_critic'):
        vae = model.actor_critic.vae
        actor = model.actor_critic.actor
    else:
        raise AttributeError("Unknown model format: expected .vae and .actor or .actor_critic")

    vae.eval()
    actor.eval()

    num_obs = vae.decoder[-1].out_features
    cenet_in_dim = vae.encoder[0].in_features
    total_input_dim = cenet_in_dim + num_obs
    num_actions = actor[-1].out_features

    print(f"num_obs={num_obs}, cenet_in_dim={cenet_in_dim}, total_dim={total_input_dim}, num_actions={num_actions}")

    class ONNXExporter(torch.nn.Module):
        def __init__(self, actor, vae):
            super().__init__()
            self.actor = actor
            self.vae = vae
            self.cenet_in_dim = vae.encoder[0].in_features
            self.num_obs = vae.decoder[-1].out_features

        def forward(self, obs_history):
            code, _, _, _ = self.vae.cenet_forward(obs_history[:, :self.cenet_in_dim])
            actor_input = torch.cat((code, obs_history[:, -self.num_obs:]), dim=1)
            return self.actor(actor_input)

    exporter = ONNXExporter(actor, vae)

    output_dir = args.output
    if output_dir is None:
        output_dir = Path(model_path).parent
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "policy.onnx")
    dummy = torch.randn(1, total_input_dim)

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
    print(f"  Input  (obs):           {total_input_dim} = {cenet_in_dim}(history) + {num_obs}(current)")
    print(f"  Output (actions):       {num_actions}")

    info_path = os.path.join(output_dir, "model_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Source model: {model_path}\n")
        f.write(f"num_obs: {num_obs}\n")
        f.write(f"cenet_in_dim (history): {cenet_in_dim}\n")
        f.write(f"total_input_dim (ONNX input): {total_input_dim}\n")
        f.write(f"num_actions: {num_actions}\n")
        f.write(f"obs_history: {total_input_dim} = history({cenet_in_dim}) + current_obs({num_obs})\n")
    print(f"Model info saved to: {info_path}")

    print("\nUsage on dog_v2_mujoco/rl_sar-ARES:")
    print(f"  Copy {onnx_path} to your policy directory")
    print(f"  Set num_observations: {num_obs}")
    print(f"  Set observations_history: [4, 3, 2, 1, 0]  (5 frames)")
    print(f"  obs_history buffer size: {total_input_dim} ({total_input_dim})")


if __name__ == "__main__":
    main()
