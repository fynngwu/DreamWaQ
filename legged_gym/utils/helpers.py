#====================== export dreamwaq policy as onnx ===============================

import os
import copy
import torch
import torch.onnx

class PolicyExporterDWAQ(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
        self.actor.eval()
        self.vae.eval()

        self.num_obs = self.vae.decoder[-1].out_features
        self.cenet_in_dim = self.vae.encoder[0].in_features
        self.total_input_dim = self.cenet_in_dim + self.num_obs

    def forward(self, obs_history):
        code, _, _, _ = self.vae.cenet_forward(obs_history[:, :self.cenet_in_dim])
        actor_input = torch.cat((code, obs_history[:, -self.num_obs:]), dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def export(self, path, opset=11):
        os.makedirs(path, exist_ok=True)
        onnx_file = os.path.join(path, 'policy_dwaq.onnx')

        dummy_obs = torch.randn(1, self.total_input_dim)

        self.to('cpu')
        self.eval()
        with torch.no_grad():
            torch.onnx.export(
                self,
                (dummy_obs,),
                onnx_file,
                input_names=['obs'],
                output_names=['actions'],
                dynamic_axes={
                    'obs': {0: 'batch'},
                    'actions': {0: 'batch'}
                },
                opset_version=opset
            )
        print(f'ONNX 模型已导出至: {onnx_file}')


def export_policy_as_dwaq(actor_critic, path):
    exporter = PolicyExporterDWAQ(actor_critic)
    exporter.export(path)


class PolicyExporterDWAQ_(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
        self.actor.eval()
        self.vae.eval()

        self.num_obs = self.vae.decoder[-1].out_features
        self.cenet_in_dim = self.vae.encoder[0].in_features
        self.total_input_dim = self.cenet_in_dim + self.num_obs

    def forward(self, obs_history):
        code, _, _, _ = self.vae.cenet_forward(obs_history[:, :self.cenet_in_dim])
        actor_input = torch.cat((code, obs_history[:, -self.num_obs:]), dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def export(self, path, opset=11):
        os.makedirs(path, exist_ok=True)
        onnx_file = os.path.join(path, 'policy_dwaq.onnx')

        dummy_obs = torch.randn(1, self.total_input_dim)

        self.to('cpu')
        self.eval()
        with torch.no_grad():
            torch.onnx.export(
                self,
                (dummy_obs,),
                onnx_file,
                input_names=['obs'],
                output_names=['actions'],
                dynamic_axes={
                    'obs': {0: 'batch'},
                    'actions': {0: 'batch'}
                },
                opset_version=opset
            )
        print(f'ONNX 模型已导出至: {onnx_file}')


def export_policy_as_dwaq_(actor_critic, path):
    exporter = PolicyExporterDWAQ_(actor_critic)
    exporter.export(path)