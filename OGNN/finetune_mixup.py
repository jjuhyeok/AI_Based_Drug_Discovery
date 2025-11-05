import os, random, warnings, argparse, glob
import numpy as np, pandas as pd, torch, dgl, torch.distributions as dist
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem.rdmolops import GetMolFrags
import numpy as np, torch, torch.distributions as dist

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric.loader import DataLoader
import torch_geometric

from dualgraph.mol import smiles2graphwithface
from dualgraph.dataset import DGData
from dualgraph.gnn import GNN2
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# -------------------- 시드 고정 --------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dgl.seed(seed); torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

# -------------------- 데이터셋 --------------------
class CustomDataset(InMemoryDataset):
    def __init__(self, root, df, mode="train", target_col="Inhibition",
                 transform=None, pre_transform=None):
        self.df, self.mode, self.target_col = df, mode, target_col
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return [f"data_{self.df.shape[0]}_{self.mode}.pt"]
    
    def raw_file_names(self):
        """PyG가 존재 여부만 확인하므로 빈 리스트로 두면 OK."""
        return []

    def download(self): pass  # not used

    def process(self):
        smiles_list = self.df["Canonical_Smiles"].values
        targets_list = self.df[[self.target_col]].values.astype(np.float32)
        data_list, skipped = [], 0

        for smi, target in tqdm(zip(smiles_list, targets_list),
                                total=len(smiles_list),
                                desc=f"Processing {self.mode}"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: skipped += 1; continue

            # 가장 큰 유기 frag 선택
            frag_smi = max(
                (Chem.MolToSmiles(f) for f in GetMolFrags(mol, asMols=True)
                 if f.HasSubstructMatch(Chem.MolFromSmarts("[#6]")) and f.GetNumAtoms() >= 3),
                key=lambda s: Chem.MolFromSmiles(s).GetNumAtoms(),
                default=None)
            if frag_smi is None: skipped += 1; continue

            graph = smiles2graphwithface(frag_smi)
            if graph is None or graph["num_nodes"] == 0: skipped += 1; continue

            d = DGData()
            d.__num_nodes__ = int(graph["num_nodes"])
            d.edge_index = torch.tensor(graph["edge_index"]).long()
            d.edge_attr = torch.tensor(graph["edge_feat"]).long()
            d.x = torch.tensor(graph["node_feat"]).long()
            d.y = torch.tensor(target, dtype=torch.float32).squeeze()
            d.ring_mask  = torch.tensor(graph["ring_mask"]).bool()
            d.ring_index = torch.tensor(graph["ring_index"]).long()
            d.nf_node    = torch.tensor(graph["nf_node"]).long()
            d.nf_ring    = torch.tensor(graph["nf_ring"]).long()
            d.num_rings  = int(graph["num_rings"])
            d.n_edges, d.n_nodes, d.n_nfs = map(int,
                (graph["n_edges"], graph["n_nodes"], graph["n_nfs"]))
            data_list.append(d)

        self.data, self.slices = self.collate(data_list)
        self.graph_list = data_list
        print(f"{self.mode}: total={len(smiles_list)}, valid={len(data_list)}, skipped={skipped}")

    def len(self):  return len(self.graph_list)
    def get(self, idx): return self.graph_list[idx]

# -------------------- 모델 --------------------
class MedModel(nn.Module):
    def __init__(self, latent=128, ckpt_path='ckpt_pretrain/ognn_pretrain_best.pt'):
        super().__init__()
        self.gnn = GNN2(mlp_hidden_size=512, mlp_layers=2, latent_size=latent,
                        use_layer_norm=False, use_face=True, ddi=True,
                        dropedge_rate=0.1, dropnode_rate=0.1, dropout=0.1,
                        dropnet=0.1, global_reducer="sum", node_reducer="sum",
                        face_reducer="sum", graph_pooling="sum",
                        node_attn=True, face_attn=True)
        self.gnn.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        self.head = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, latent),
            nn.BatchNorm1d(latent),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(latent, 1),
        )
        self.head[-1].weight.data.normal_(0.0, 0.01)

    def get_embedding(self, batch):
        return self.gnn(batch) if batch.x.numel() else torch.empty(batch.num_graphs,
                                                                   self.gnn.latent_size,
                                                                   device=batch.y.device)
    def forward(self, batch):
        return self.head(self.get_embedding(batch)).squeeze(1).clamp(0, 100)

# -------------------- Loss / Metric --------------------
def calc_lb(y_true, y_pred):
    y_true, y_pred = map(np.asarray, (y_true, y_pred))
    msk = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if msk.sum() < 2: return 0, 0, 0
    rmse = np.sqrt(mean_squared_error(y_true[msk], y_pred[msk]))
    A = rmse / (y_true[msk].max() - y_true[msk].min() + 1e-8)
    B = np.clip(np.corrcoef(y_true[msk], y_pred[msk])[0, 1], 0, 1)
    return rmse, B, 0.5*(1 - min(A,1)) + 0.5*B

class LBLoss(nn.Module):
    def __init__(self, a=0.5, b=0.5):
        super().__init__(); self.a, self.b = a, b; self.mse = nn.MSELoss()
    def forward(self, p, t):
        mse = self.mse(p, t); rmse = torch.sqrt(mse)
        rng = (t.max() - t.min()).clamp_min(1e-8)
        A = (rmse / rng).clamp(max=1.0)
        vx, vy = p - p.mean(), t - t.mean()
        B = torch.clamp((vx*vy).sum() / (vx.norm()*vy.norm()+1e-8), 0, 1)
        return self.a*A + self.b*(1 - B)

def embed_sim_matrix(h: torch.Tensor) -> torch.Tensor:
    """
    h : [n, d]  (그래프 임베딩)
    반환값 : [n, n]  코사인 유사도 행렬을 0~1 범위로 매핑
    """
    h_norm = F.normalize(h.detach(), dim=1)          # 단위 벡터화 & 그래프 분리
    sim    = torch.mm(h_norm, h_norm.t())            # [-1, 1]
    return (sim + 1) / 2                             # → [0, 1]


def sample_mixup_embed(
        h: torch.Tensor,               # [n, d] 그래프 임베딩
        y: torch.Tensor,               # [n]    라벨
        alpha: float = 0.4,
        w_label: float = 0.5,
        w_embed: float = 0.5):
    """
    라벨·임베딩 유사도 가중 Manifold Mixup
    (Tanimoto 대신 코사인 유사도 사용)
    """
    n = h.size(0)
    if n < 2:
        return h, y                    # 배치가 1개면 Mixup 생략

    # 1) 라벨 유사도
    y_dist  = torch.cdist(y.unsqueeze(1), y.unsqueeze(1), p=1)
    label_sim = (1 - y_dist / y_dist[y_dist != 0].max().clamp_min(1e-8)) \
                if (y_dist != 0).any() else torch.zeros_like(y_dist)

    # 2) 임베딩 유사도(코사인)
    embed_sim = embed_sim_matrix(h).to(h.device)

    # 3) 혼합 확률
    sim  = w_label * label_sim + w_embed * embed_sim
    base = (1 - (w_label + w_embed)) / (n - 1)
    prob = base + sim
    prob.fill_diagonal_(0)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # 4) 파트너 선택 & 보간
    perm  = torch.multinomial(prob, 1).squeeze()
    lam   = dist.Beta(alpha, alpha).sample((n, 1)).to(h.device)
    h_mix = lam * h + (1 - lam) * h[perm]
    y_mix = (lam.squeeze() * y) + ((1 - lam.squeeze()) * y[perm])
    return h_mix, y_mix


# -------------------- 학습 --------------------
def main(cfg):
    seed_everything(cfg.seed)
    base = '../OGNN'

    train_df = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
    test_df  = pd.read_csv(os.path.join(base, 'data', 'test.csv'))
    test_df['Inhibition'] = 0

    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=cfg.seed)
    ids = np.arange(len(train_df)); fold = cfg.fold - 1
    tr_idx, va_idx = list(kf.split(ids))[fold]
    tr_df, va_df   = train_df.iloc[tr_idx], train_df.iloc[va_idx]

    ds_tr = CustomDataset(root=f'{base}/proc_tr', df=tr_df, mode='train')
    ds_va = CustomDataset(root=f'{base}/proc_va', df=va_df, mode='valid')
    ds_te = CustomDataset(root=f'{base}/proc_te', df=test_df, mode='test')
    dl_tr = DataLoader(ds_tr, 128, True, worker_init_fn=seed_worker)
    dl_va = DataLoader(ds_va, 128, False)
    dl_te = DataLoader(ds_te, 128, False)

    model = MedModel().to(cfg.device)
    loss_fn = LBLoss(); opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10)

    beta = dist.Beta(cfg.mixup_alpha, cfg.mixup_alpha)
    best = -1.0; save_dir = os.path.join(base, cfg.save_path); os.makedirs(save_dir, exist_ok=True)

    for ep in range(100):
        model.train(); tr_loss = 0; n_batch = 0
        for batch in tqdm(dl_tr, desc=f"Ep{ep:02d}"):
            batch = batch.to(cfg.device)
            h = model.get_embedding(batch); y = batch.y
            h_mix, y_mix = sample_mixup_embed(
                h, y,
                alpha = cfg.mixup_alpha,
                w_label = 1 - cfg.struct_weight,
                w_embed = cfg.struct_weight)
            pred = model.head(h_mix).squeeze(1).clamp(0,100)
            loss = loss_fn(pred, y_mix); opt.zero_grad(); loss.backward()
            clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update()
            tr_loss += loss.item(); n_batch += 1
        sched.step()

        # ------- Validation -------
        model.eval(); ema.store(); ema.copy_to(model.parameters())
        preds, labels = [], []
        with torch.no_grad():
            for b in dl_va:
                b = b.to(cfg.device)
                preds.extend(model(b).cpu().numpy())
                labels.extend(b.y.cpu().numpy())
        rmse, R, lb = calc_lb(labels, preds)
        ema.restore()

        print(f"E{ep:02d}  train_loss={tr_loss/n_batch:.4f}  RMSE={rmse:.3f}  R={R:.3f}  LB={lb:.4f}  best={best:.4f}")
        if lb > best:
            best = lb
            torch.save(model.state_dict(), f"{save_dir}/{cfg.seed}_{cfg.fold}_best.pt")
            # -------- Test inference --------
            out = []
            with torch.no_grad():
                for b in dl_te:
                    b = b.to(cfg.device); out.extend(model(b).cpu().numpy())
            np.save(f"{save_dir}/test_{cfg.seed}_fold{cfg.fold}_{best:.4f}.npy", np.asarray(out))

# -------------------- CLI --------------------
def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--label_similarity_weight", type=float, default=0.5)
    p.add_argument("--save_path", type=str, default="results")
    p.add_argument("--struct_weight", type=float, default=0.3,
               help="임베딩(구조) 유사도 가중치 (0~1). 0이면 라벨 유사도만 사용")


    return p.parse_args()

if __name__ == "__main__":
    cfg = parse_cfg(); main(cfg)
