import joblib, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors # 이 모듈은 더 이상 직접적으로 사용되지 않지만, 다른 곳에서 필요할 수 있어 일단 유지합니다.
from rdkit.Chem.rdmolops import GetMolFrags

import argparse
import torch, os, random
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch_ema import ExponentialMovingAverage
from matplotlib import pyplot as plt
from glob import glob
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GNN2
from torch.nn.utils import clip_grad_norm_
import torch.distributions as dist

import torch_geometric
from torch_geometric.data import Dataset, InMemoryDataset, Batch, Data
from dualgraph.dataset import DGData
from torch_geometric.loader import DataLoader
import dgl
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, target_type='Inhibition', mode='train'):
        self.df = df
        self.target_type = target_type
        self.mode = mode
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self.df.shape[0]}_{self.mode}.pt']

    def download(self):
        pass

    def process(self):
        from rdkit import Chem
        from rdkit.Chem.rdmolops import GetMolFrags

        smiles_list = self.df["Canonical_Smiles"].values
        targets_list = self.df[['Inhibition']].values.astype(np.float32)
        data_list = []
        skipped_count = 0
        valid_count = 0

        for i in tqdm(range(len(smiles_list)), desc=f"Processing {self.mode} data"):
            smi = smiles_list[i]
            targets = targets_list[i]
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped_count += 1
                continue

            frags = GetMolFrags(mol, asMols=True)
            selected_smi = None
            max_organic_atoms = 0

            for frag in frags:
                if frag.HasSubstructMatch(Chem.MolFromSmarts("[#6]")):
                    if frag.GetNumAtoms() >= 3:
                        if frag.GetNumAtoms() > max_organic_atoms:
                            max_organic_atoms = frag.GetNumAtoms()
                            selected_smi = Chem.MolToSmiles(frag)
            
            if selected_smi is None:
                skipped_count += 1
                continue

            graph = smiles2graphwithface(selected_smi)

            if graph is None or \
               graph.get("num_nodes", 0) == 0 or \
               graph.get("edge_index", np.array([])).size == 0 or \
               graph.get("node_feat", np.array([])).size == 0:
                skipped_count += 1
                continue

            data = DGData()
            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).long()
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).long()
            data.x = torch.from_numpy(graph["node_feat"]).long()
            
            data.y = torch.tensor(targets, dtype=torch.float32).squeeze()
            
            data.ring_mask = torch.from_numpy(graph["ring_mask"]).bool()
            data.ring_index = torch.from_numpy(graph["ring_index"]).long()
            data.nf_node = torch.from_numpy(graph["nf_node"]).long()
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).long()
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            
            data_list.append(data)
            valid_count += 1

        print(f"\n--- CustomDataset {self.mode} Build Summary ---")
        print(f"Total SMILES count: {len(smiles_list)}")
        print(f"Valid graph count (added to data_list): {valid_count}")
        print(f"Skipped graph count: {skipped_count}")
        if len(smiles_list) > 0:
            print(f"Valid ratio: {valid_count / len(smiles_list):.2f}\n")
        else:
            print(f"Valid ratio: N/A (Total SMILES count is 0.)\n")

        if data_list:
            self.data, self.slices = self.collate(data_list)
        else:
            self.data = Batch.from_data_list([])
            self.slices = {key: torch.empty(0, dtype=torch.long) for key in data_list[0].keys()} if data_list else {}
        
        self.graph_list = data_list
        
    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        data = self.graph_list[idx]
        return data

class MedModel(torch.nn.Module):
    def __init__(self, latent_size=128): # latent_size를 인자로 받도록 수정
        super(MedModel, self).__init__()
        self.ddi = True
        self.gnn = GNN2(
            mlp_hidden_size=512, mlp_layers=2, latent_size=latent_size, # latent_size 사용
            use_layer_norm=False, use_face=True, ddi=self.ddi,
            dropedge_rate=0.1, dropnode_rate=0.1, dropout=0.1,
            dropnet=0.1, global_reducer="sum", node_reducer="sum",
            face_reducer="sum", graph_pooling="sum",
            node_attn=True, face_attn=True
        )
        pretrain_ckpt_path = os.path.join('/content/drive/MyDrive/OGNN', 'ckpt_pretrain', 'ognn_pretrain_best.pt')
        
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(f"Pre-trained model checkpoint not found at: {pretrain_ckpt_path}")

        state_dict = torch.load(pretrain_ckpt_path, map_location='cpu')
        self.gnn.load_state_dict(state_dict)

        self.fc1 = nn.Sequential(
            nn.LayerNorm(latent_size), # latent_size 사용
            nn.Linear(latent_size, latent_size), # latent_size 사용
            nn.BatchNorm1d(latent_size), # latent_size 사용
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(latent_size, 1), # latent_size 사용
        )
        self.fc1[-1].weight.data.normal_(mean=0.0, std=0.01)

    def get_embedding(self, batch):
        if batch.x is None or batch.x.numel() == 0:
            return torch.empty(batch.num_graphs, self.gnn.latent_size, device=batch.y.device)
        return self.gnn(batch)

    def predict_from_embedding(self, mol_embedding):
        out = self.fc1(mol_embedding).squeeze(1)
        # Ensure output is finite after prediction
        if torch.isnan(out).any() or torch.isinf(out).any():
            warnings.warn("Model prediction contains NaN or Inf values. Clamping to 0-100.")
            out = torch.clamp(out.nan_to_num(nan=0.0, posinf=100.0, neginf=0.0), 0, 100) # Use nan_to_num
        else:
            out = torch.clamp(out, 0, 100)
        return out


    def forward(self, batch):
        mol_embedding = self.get_embedding(batch)
        return self.predict_from_embedding(mol_embedding)


def correlation_score(y_true, y_pred):
    # Convert to numpy for robust checks before torch operations
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # Filter out NaNs if they somehow get here, though they should be handled earlier
    valid_mask = ~np.isnan(y_true_np) & ~np.isnan(y_pred_np)
    y_true_np_filtered = y_true_np[valid_mask]
    y_pred_np_filtered = y_pred_np[valid_mask]

    if len(y_true_np_filtered) < 2: # Need at least 2 points for correlation
        return torch.tensor(0.0, device=y_true.device)

    y_true_centered = torch.from_numpy(y_true_np_filtered).to(y_true.device) - torch.mean(torch.from_numpy(y_true_np_filtered).to(y_true.device))
    y_pred_centered = torch.from_numpy(y_pred_np_filtered).to(y_pred.device) - torch.mean(torch.from_numpy(y_pred_np_filtered).to(y_pred.device))
    
    cov_tp = torch.sum(y_true_centered * y_pred_centered)
    var_t = torch.sum(y_true_centered ** 2)
    var_p = torch.sum(y_pred_centered ** 2)
    
    denom = torch.sqrt(var_t * var_p) + 1e-8
    
    return cov_tp / denom


def correlation_loss(pred, target):
    return -correlation_score(target, pred)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--save_path", default='results', type=str)        
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--mixup_alpha", type=float, default=0.4, help="Alpha parameter for Beta distribution in Mixup.")
    # 바닐라 Mixup 기반이므로 knn_neighbors는 제거합니다.
    # 레이블 유사성 가중치 인자는 유지합니다.
    parser.add_argument("--label_similarity_weight", type=float, default=0.5, 
                        help="Weight for label similarity in sampling (0.0 for pure random, 1.0 for strong label influence).")
    args = parser.parse_args()

    return args

def calculate_lb_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out NaN values from both arrays simultaneously
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0, 0.0, 0.0 # rmse, pearson, score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        normalized_rmse = 0.0
    else:
        normalized_rmse = rmse / y_range

    A = normalized_rmse

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_coeff = 0.0
    else:
        pearson_coeff = np.corrcoef(y_true, y_pred)[0, 1]
    
    B = np.clip(pearson_coeff, 0, 1)

    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    
    return rmse, pearson_coeff, score

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        if preds.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        if preds.ndim == 0 or targets.ndim == 0:
             return torch.tensor(0.0, device=preds.device, requires_grad=True)

        preds = preds.squeeze() if preds.ndim > 1 else preds
        targets = targets.squeeze() if targets.ndim > 1 else targets

        if preds.shape != targets.shape:
            warnings.warn(f"Shape mismatch in CustomLoss: preds {preds.shape} vs targets {targets.shape}")
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Check for NaNs/Infs in preds and targets BEFORE loss calculation
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            warnings.warn("NaN or Inf detected in predictions before loss. Clamping to finite values.")
            preds = preds.nan_to_num(nan=0.0, posinf=100.0, neginf=0.0) # Convert NaNs/Infs to finite values
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            warnings.warn("NaN or Inf detected in targets before loss. Clamping to finite values.")
            targets = targets.nan_to_num(nan=0.0, posinf=100.0, neginf=0.0) # Convert NaNs/Infs to finite values


        mse = self.mse_loss(preds, targets)
        rmse = torch.sqrt(mse)

        min_t = torch.min(targets)
        max_t = torch.max(targets)
        
        range_t = (max_t - min_t)
        norm_rmse = rmse / (range_t + 1e-8)

        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)
        
        denom_corr = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
        corr = torch.sum(vx * vy) / denom_corr
        corr_clipped = torch.clamp(corr, 0, 1)

        loss = self.alpha * torch.min(norm_rmse, torch.tensor(1.0, device=norm_rmse.device)) + self.beta * (1 - corr_clipped)
        return loss


def main(args):
    seed_everything(args.seed)
    
    base_dir = '/content/drive/MyDrive/OGNN'
    data_path = os.path.join(base_dir, 'data', 'train.csv')
    test_data_path = os.path.join(base_dir, 'data', 'test.csv')

    data = pd.read_csv(data_path, index_col=None)
    data['ID'] = data.index.values

    test = pd.read_csv(test_data_path, index_col=None)
    test['Inhibition'] = 0
    test['ID'] = test.index.values

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    ids = data['ID'].values

    fold = args.fold - 1
    t_ids, v_ids = None, None
    for i, (train_index, val_index) in enumerate(kf.split(ids)):
        if i == fold:
            t_ids, v_ids = ids[train_index], ids[val_index]
            break

    train = data[data['ID'].isin(t_ids)].reset_index(drop=True)
    valid = data[data['ID'].isin(v_ids)].reset_index(drop=True)

    print("Building train dataset...")
    train_dataset = CustomDataset(df=train, mode='train', root=os.path.join(base_dir, 'processed_data_train'))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, worker_init_fn=seed_worker)

    print("Building valid dataset...")
    valid_dataset = CustomDataset(df=valid, mode='test', root=os.path.join(base_dir, 'processed_data_valid'))
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=0)

    print("Building test dataset...")
    test_dataset = CustomDataset(df=test, mode='test', target_type='Inhibition', root=os.path.join(base_dir, 'processed_data_test'))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = MedModel().to(args.device)
    criterion = CustomLoss(alpha=0.5, beta=0.5)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=1, verbose=False)
    
    best_val_lb_score = -1.0 # Best LB Score tracking

    model_save_dir = os.path.join(base_dir, args.save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    print(f"Starting training on device: {args.device}")
    print(f"Models will be saved to: {model_save_dir}")

    mixup_beta_dist = dist.Beta(args.mixup_alpha, args.mixup_alpha)

    for epoch in range(45):
        model.train()
        train_loss_total = 0
        train_batches_processed = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch} Train")):
            batch = batch.to(args.device)
            
            if batch.x is None or batch.x.numel() == 0:
                warnings.warn("Skipping batch: batch.x is None or empty.")
                continue

            # --- Label Similarity based Vanilla Mixup augmentation start ---
            # 1. Pre-calculate embeddings for the current batch
            current_batch_embeddings = model.get_embedding(batch)
            
            # Check for NaN/Inf in embeddings
            if torch.isnan(current_batch_embeddings).any() or torch.isinf(current_batch_embeddings).any():
                warnings.warn(f"Ep {epoch} Batch {batch_idx}: current_batch_embeddings contains NaN/Inf. Skipping Mixup for this batch.")
                # Fallback to pure prediction if embeddings are bad
                pred = model.predict_from_embedding(current_batch_embeddings.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0))
                loss = criterion(pred, batch.y)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Ep {epoch} Training Batch {batch_idx}: Loss is NaN or Inf. Skipping this batch. Loss: {loss.item()}")
                    optim.zero_grad()
                    continue
                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                ema.update()
                train_loss_total += loss.item()
                train_batches_processed += 1
                continue

            targets_np = batch.y.detach().cpu().numpy()
            n_samples_in_batch = batch.num_graphs

            mixed_mol_embeddings_list = []
            mixed_targets_list = []

            if n_samples_in_batch < 2: # Cannot perform mixup with less than 2 samples
                pred = model.predict_from_embedding(current_batch_embeddings)
                loss = criterion(pred, batch.y)
                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                ema.update()
                train_loss_total += loss.item()
                train_batches_processed += 1
                continue

            for i in range(n_samples_in_batch):
                # Calculate label distances for all other samples in the batch
                label_distances = np.abs(targets_np[i] - targets_np)
                
                # Exclude self (distance to self is 0, which would give infinite probability)
                label_distances[i] = np.inf # Exclude self from consideration

                # Normalize label distances to 0-1 range based on batch min/max
                max_label_dist = np.max(targets_np) - np.min(targets_np)
                
                probabilities = None
                if max_label_dist == 0 or (n_samples_in_batch - 1) == 0: # Handle edge case where n_samples_in_batch is 1
                    # If all targets in the batch are the same OR batch size is 1, fall back to pure random (except self)
                    probabilities = np.ones(n_samples_in_batch)
                    probabilities[i] = 0 # Exclude self
                    probabilities = probabilities / (np.sum(probabilities) + 1e-8)
                else:
                    normalized_label_distances = label_distances / max_label_dist
                    
                    # Similarity is higher when distance is lower
                    # We want to give higher probability to samples with high similarity (low distance)
                    similarity_scores = (1.0 - normalized_label_distances) # 0 for max dist, 1 for min dist (except self)
                    
                    # Combine a uniform distribution with the similarity-weighted distribution
                    # A higher `label_similarity_weight` means more emphasis on similarity.
                    # A lower `label_similarity_weight` (closer to 0) means more emphasis on uniform random sampling.
                    
                    # Uniform base probability (for the "vanilla" component)
                    # Correctly initialize as an array
                    uniform_prob_val = (1.0 - args.label_similarity_weight) / (n_samples_in_batch - 1 + 1e-8)
                    initial_probabilities = np.full(n_samples_in_batch, uniform_prob_val)
                    initial_probabilities[i] = 0.0 # Ensure self is 0

                    # Similarity-weighted probability component
                    # We can scale similarity_scores directly by args.label_similarity_weight
                    weighted_similarity_probs = args.label_similarity_weight * similarity_scores
                    weighted_similarity_probs[i] = 0.0 # Self has 0 probability

                    probabilities = initial_probabilities + weighted_similarity_probs
                    
                    # Final normalization to ensure probabilities sum to 1
                    probabilities = probabilities / (np.sum(probabilities) + 1e-8)

                # Select a neighbor based on these probabilities
                if np.sum(probabilities) == 0: # Fallback if normalization resulted in all zeros
                    j = torch.randint(0, n_samples_in_batch, (1,)).item()
                    while j == i:
                        j = torch.randint(0, n_samples_in_batch, (1,)).item()
                else:
                    try:
                        j = np.random.choice(np.arange(n_samples_in_batch), p=probabilities)
                    except ValueError as e:
                        warnings.warn(f"Probability distribution error: {e}. Falling back to random choice. Probs: {probabilities}")
                        j = torch.randint(0, n_samples_in_batch, (1,)).item()
                        while j == i:
                            j = torch.randint(0, n_samples_in_batch, (1,)).item()
                
                # Sample lambda value
                lambda_val = mixup_beta_dist.sample((1, 1)).to(args.device)

                # Apply Manifold Mixup
                mol_embedding_i = current_batch_embeddings[i:i+1] # Keep dim
                mol_embedding_j = current_batch_embeddings[j:j+1] # Keep dim

                mixed_mol_embedding = lambda_val * mol_embedding_i + (1 - lambda_val) * mol_embedding_j
                mixed_mol_embeddings_list.append(mixed_mol_embedding)

                target_original_i = batch.y[i:i+1].to(args.device)
                target_shuffled_j = batch.y[j:j+1].to(args.device)

                mixed_target = lambda_val.squeeze() * target_original_i + (1 - lambda_val.squeeze()) * target_shuffled_j
                mixed_targets_list.append(mixed_target)

            # Concatenate lists into tensors
            mixed_mol_embedding_final = torch.cat(mixed_mol_embeddings_list, dim=0)
            mixed_target_final = torch.cat(mixed_targets_list, dim=0)
            # --- Label Similarity based Vanilla Mixup augmentation end ---
            
            pred = model.predict_from_embedding(mixed_mol_embedding_final)
            
            # Additional check for NaNs in pred before loss
            if pred.numel() == 0 or mixed_target_final.numel() == 0 or \
               torch.isnan(pred).any() or torch.isinf(pred).any() or \
               torch.isnan(mixed_target_final).any() or torch.isinf(mixed_target_final).any():
                warnings.warn(f"Ep {epoch} Batch {batch_idx}: Prediction or target contains NaN/Inf or is empty after Mixup. Skipping loss calculation for this batch.")
                continue
            
            if pred.shape != mixed_target_final.shape:
                warnings.warn(f"Shape mismatch before loss: pred {pred.shape} vs target {mixed_target_final.shape}. Skipping batch.")
                continue

            loss = criterion(pred, mixed_target_final)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Ep {epoch} Training Batch {batch_idx}: Loss is NaN or Inf. Skipping this batch. Loss: {loss.item()}")
                optim.zero_grad()
                continue

            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0) # Lower clip value
            optim.step()
            ema.update()

            train_loss_total += loss.item()
            train_batches_processed += 1
            
        avg_train_loss = train_loss_total / train_batches_processed if train_batches_processed > 0 else 0.0

        model.eval()
        valid_preds, valid_labels = [], []
        for batch_val_idx, batch in enumerate(tqdm(valid_loader, desc=f"Ep {epoch} Valid")):
            batch = batch.to(args.device)
            
            if batch.x is None or batch.x.numel() == 0:
                continue

            with torch.no_grad():
                pred = model(batch)
                target = batch.y.to(args.device)
                
                # Filter out NaNs/Infs at the point of collection for evaluation
                pred_cpu = pred.cpu().numpy()
                target_cpu = target.cpu().numpy()

                valid_mask = ~np.isnan(pred_cpu) & ~np.isinf(pred_cpu) & \
                             ~np.isnan(target_cpu) & ~np.isinf(target_cpu)

                if np.any(~valid_mask):
                    warnings.warn(f"Ep {epoch} Validation Batch {batch_val_idx}: Detected NaN/Inf in predictions or targets. Filtering {np.sum(~valid_mask)} samples.")

                valid_preds.extend(pred_cpu[valid_mask].tolist())
                valid_labels.extend(target_cpu[valid_mask].tolist())
        
        # --- Calculate and print validation metrics ---
        val_rmse, val_pearson, val_lb_score = 0.0, 0.0, 0.0
        if len(valid_labels) > 1 and len(valid_preds) > 1:
            val_rmse, val_pearson, val_lb_score = calculate_lb_score(valid_labels, valid_preds)
        else:
            print(f"Warning: Ep {epoch} Validation data has insufficient valid samples to calculate metrics. (Sample count: {len(valid_labels)})")

        scheduler.step()

        print(f'EPOCH : {epoch:02d} | T_LOSS : {avg_train_loss:.4f} | VAL_RMSE : {val_rmse:.4f} | VAL_PEARSON : {val_pearson:.4f} | VAL_LB_SCORE : {val_lb_score:.4f} | BEST_VAL_LB_SCORE : {best_val_lb_score:.4f}')
        
        if val_lb_score > best_val_lb_score:
            best_val_lb_score = val_lb_score
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'{args.seed}_{args.fold}.pt'))

            # --- Test set prediction and saving (only for best LB Score) ---
            ognn_MLM = []
            for batch in tqdm(test_loader, desc=f"Ep {epoch} Test"):
                batch = batch.to(args.device)
                if batch.x is None or batch.x.numel() == 0:
                    continue
                with torch.no_grad():
                    pred_mlm = model(batch)
                    
                    # Filter out NaNs/Infs from test predictions before saving
                    pred_mlm_cpu = pred_mlm.cpu().numpy()
                    valid_mask_test = ~np.isnan(pred_mlm_cpu) & ~np.isinf(pred_mlm_cpu)

                    if np.any(~valid_mask_test):
                            warnings.warn(f"Ep {epoch} Test Batch: Detected NaN/Inf in test predictions. Filtering {np.sum(~valid_mask_test)} samples.")
                    
                    ognn_MLM.extend(pred_mlm_cpu[valid_mask_test].tolist()) # Only extend valid predictions

            ognn_MLM = np.array(ognn_MLM)

            for f in glob(os.path.join(model_save_dir, f'test_{args.seed}_*_fold{args.fold}*.npy')):
                os.remove(f)
            np.save(os.path.join(model_save_dir, f'test_{args.seed}_Inhibition_fold{args.fold}_{best_val_lb_score:.4f}.npy'), ognn_MLM)

if __name__ == '__main__':
    args = parse_args()
    main(args)