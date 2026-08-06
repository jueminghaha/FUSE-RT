"""UniRT-GINKAN architecture used by the external OOD baseline.

Upstream implementation: https://github.com/hcji/Uni-RT
Reproducibility reference: b981fc118b1264f937626b8ec980b426100537b1
Uni-RT is distributed under the MIT License; this module keeps the upstream
provenance visible while adapting paths and evaluation entry points locally.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, GINConv, global_add_pool

CUDA_DEVICE_INDEX = int(os.environ.get("CUDA_DEVICE_INDEX", "0"))
DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu")
CFG = {
    "mode": "RPLC",
    "model_type": "GINKAN",
    "hidden": 64,
    "layers": 4,
    "dropout": 0.02,
    "adapter_reduction": 4,
    "use_cross_stitch": True,
    "task_emb_dim_mode": "num_tasks",
}

# Cell 3 — reproducibility and molecule featurization copied/ported from Uni-RT scripts

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ATOMIC_NUM_SET = list(range(1, 40))
HYBRIDIZATION_SET = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPE_SET = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

def one_hot(x, allowed):
    return [int(x == s) for s in allowed] + [int(x not in allowed)]

def atom_features(atom: rdchem.Atom) -> List[int]:
    return (
        one_hot(atom.GetAtomicNum(), ATOMIC_NUM_SET)
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_SET)
        + [
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED),
        ]
    )

def bond_features(bond: Optional[rdchem.Bond]) -> List[int]:
    if bond is None:
        return [0] * (len(BOND_TYPE_SET) + 1 + 3)
    bt = bond.GetBondType()
    return (
        one_hot(bt, BOND_TYPE_SET)
        + [
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo() != rdchem.BondStereo.STEREONONE),
        ]
    )

def smiles_to_data(smiles: str, y: Optional[float], task_id: Optional[int]) -> Data:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        for idx in range(mol.GetNumAtoms()):
            edge_index.append([idx, idx])
            edge_attr.append(bond_features(None))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None and not (isinstance(y, float) and math.isnan(y)):
        data.y = torch.tensor([y], dtype=torch.float)
    else:
        data.y = torch.tensor([float("nan")], dtype=torch.float)
    data.smiles = smiles
    if task_id is not None:
        data.task = torch.tensor([int(task_id)], dtype=torch.long)
    return data

node_dim = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
print("Node feature dimension:", node_dim)
print("Bond feature dimension:", len(bond_features(None)))

# Cell 4 — model classes copied/ported from Uni-RT scripts

class FiLMParamGenerator(nn.Module):
    def __init__(self, task_emb_dim: int, hidden: int, scale_gamma: float = 0.5, scale_beta: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_emb_dim, task_emb_dim),
            nn.ReLU(),
            nn.Linear(task_emb_dim, 2 * hidden),
        )
        self.scale_gamma = nn.Parameter(torch.tensor(scale_gamma), requires_grad=False)
        self.scale_beta = nn.Parameter(torch.tensor(scale_beta), requires_grad=False)

    def forward(self, task_emb: torch.Tensor):
        out = self.net(task_emb)
        gamma_raw, beta_raw = out.chunk(2, dim=-1)
        gamma = 1.0 + self.scale_gamma * torch.tanh(gamma_raw)
        beta = self.scale_beta * torch.tanh(beta_raw)
        return gamma, beta

class Adapter(nn.Module):
    def __init__(self, hidden_dim: int, reduction: int = 4):
        super().__init__()
        self.down = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.up = nn.Linear(hidden_dim // reduction, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, h):
        return self.up(self.relu(self.down(h))) + h

class CrossStitch(nn.Module):
    def __init__(self, num_tasks: int, main_task_id: int = 0, alpha_init: float = 1.0, beta_init: float = 0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.main_task_id = main_task_id
        alpha = torch.eye(num_tasks) * alpha_init
        alpha[main_task_id, :] = beta_init
        alpha[main_task_id, main_task_id] = alpha_init
        self.alpha = nn.Parameter(alpha)

    def forward(self, h_task: torch.Tensor, graph_tasks: torch.Tensor):
        h_new = torch.zeros_like(h_task)
        for i in range(self.num_tasks):
            mask_i = graph_tasks == i
            if mask_i.sum() == 0:
                continue
            h_i = h_task[mask_i]
            if i == self.main_task_id:
                fused = 0.0
                for j in range(self.num_tasks):
                    mask_j = graph_tasks == j
                    if mask_j.sum() == 0:
                        continue
                    h_j = h_task[mask_j].mean(dim=0, keepdim=True)
                    fused = fused + self.alpha[i, j] * h_j
                h_new[mask_i] = h_i + fused
            else:
                h_new[mask_i] = h_i
        return h_new

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-12)
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)] + 1e-12)
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1).to(A.device)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        for layer in self.layers:
            # The public training scripts do not update grids during normal training.
            x = layer(x)
        return x

class MLPWithFiLM(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, film_generator_1, film_generator_2, dropout=0.0):
        super().__init__()
        self.film_generator_1 = film_generator_1
        self.film_generator_2 = film_generator_2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x, task_emb):
        gamma1, beta1 = self.film_generator_1(task_emb)
        x = gamma1 * x + beta1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        gamma2, beta2 = self.film_generator_2(task_emb)
        x = gamma2 * x + beta2
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class GINKANConv(GINConv):
    def __init__(self, input_dim, hidden_dim, output_dim, grid_size=5, spline_order=3, eps=0.0, train_eps=False):
        nn_kan = KAN([input_dim, hidden_dim, output_dim], grid_size=grid_size, spline_order=spline_order)
        super().__init__(nn=nn_kan, eps=eps, train_eps=train_eps)

class GINKANmultiRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_tasks: int = 28,
        task_emb_dim: int = 32,
        hidden: int = 64,
        layers: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.1,
        adapter_reduction: int = 4,
        use_cross_stitch: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers_n = layers
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            conv = GINKANConv(hidden, hidden, hidden, grid_size=grid_size, spline_order=spline_order)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)
        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)
        self.head = MLPWithFiLM(hidden, hidden, 1, self.head_film_gen_1, self.head_film_gen_2, dropout=dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = graph_tasks == t_id
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted
        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)
        task_emb = self.task_embed(graph_tasks)
        out = self.head(g, task_emb)
        return out.view(-1)

class GINmultiRegressor(nn.Module):
    def __init__(self, node_dim, num_tasks=28, task_emb_dim=32, hidden=64, layers=4, dropout=0.1, adapter_reduction=4, use_cross_stitch=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            nn_node = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(nn=nn_node))
            self.bns.append(BatchNorm(hidden))
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)
        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)
        self.head = MLPWithFiLM(hidden, hidden, 1, self.head_film_gen_1, self.head_film_gen_2, dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = graph_tasks == t_id
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted
        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)
        task_emb = self.task_embed(graph_tasks)
        return self.head(g, task_emb).view(-1)

class GCNmultiRegressor(nn.Module):
    def __init__(self, node_dim, num_tasks=28, task_emb_dim=32, hidden=64, layers=4, dropout=0.1, adapter_reduction=4, use_cross_stitch=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers)])
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)
        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)
        self.head = MLPWithFiLM(hidden, hidden, 1, self.head_film_gen_1, self.head_film_gen_2, dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
        g = global_add_pool(h, batch)
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = graph_tasks == t_id
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted
        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)
        task_emb = self.task_embed(graph_tasks)
        return self.head(g, task_emb).view(-1)

class GATmultiRegressor(nn.Module):
    def __init__(self, node_dim, num_tasks=28, task_emb_dim=32, hidden=64, layers=4, heads=4, dropout=0.1, adapter_reduction=4, use_cross_stitch=True):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, heads=heads, concat=True) for _ in range(layers)])
        self.bns = nn.ModuleList([BatchNorm(hidden) for _ in range(layers)])
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)
        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)
        self.head = MLPWithFiLM(hidden, hidden, 1, self.head_film_gen_1, self.head_film_gen_2, dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = F.elu(bn(conv(h, edge_index)))
        g = global_add_pool(h, batch)
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = graph_tasks == t_id
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted
        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)
        task_emb = self.task_embed(graph_tasks)
        return self.head(g, task_emb).view(-1)

def build_model(model_type: str, node_dim: int, num_tasks: int):
    task_emb_dim = num_tasks if CFG["task_emb_dim_mode"] == "num_tasks" else int(CFG["task_emb_dim_mode"])
    common = dict(
        node_dim=node_dim,
        num_tasks=num_tasks,
        task_emb_dim=task_emb_dim,
        hidden=CFG["hidden"],
        layers=CFG["layers"],
        dropout=CFG["dropout"],
        adapter_reduction=CFG["adapter_reduction"],
        use_cross_stitch=CFG["use_cross_stitch"],
    )
    model_type = model_type.upper()
    if model_type == "GINKAN":
        return GINKANmultiRegressor(**common)
    if model_type == "GIN":
        return GINmultiRegressor(**common)
    if model_type == "GCN":
        return GCNmultiRegressor(**common)
    if model_type == "GAT":
        return GATmultiRegressor(**common)
    raise ValueError(f"Unknown model_type: {model_type}")

print("Model classes loaded.")
