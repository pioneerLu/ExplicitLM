import numpy as np
import faiss


def train_kmeans(
    data: np.ndarray,
    k: int,
    batch_size: int = 10000,   # 接口保留
    max_iter: int = 100
):
    """
    Train KMeans using FAISS (GPU if available).

    - 接口保持不变
    - 避免全量 astype / contiguous
    - 仅对小规模子样本做 float32
    """
    assert data.ndim == 2, "data must be [N, dim]"
    n, d = data.shape

    # ---------- 1. 选择训练子样本（关键） ----------
    max_train_samples = min(n, max(100_000, k * 256))
    if n > max_train_samples:
        idx = np.random.choice(n, max_train_samples, replace=False)
        train_data = data[idx]
    else:
        train_data = data

    # 仅对训练数据转 float32（一次拷贝）
    train_data = train_data.astype(np.float32, copy=False)

    # ---------- 2. 是否使用 GPU ----------
    use_gpu = faiss.get_num_gpus() > 0

    # ---------- 3. 训练 KMeans ----------
    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=max_iter,
        verbose=True,
        gpu=use_gpu,
        spherical=False,
        min_points_per_centroid=5,
        max_points_per_centroid=10_000_000,
    )

    kmeans.train(train_data)

    # ---------- 4. assignment（分批，避免 OOM） ----------
    labels = np.empty(n, dtype=np.int64)
    index = kmeans.index

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = data[start:end].astype(np.float32, copy=False)
        _, lab = index.search(batch, 1)
        labels[start:end] = lab[:, 0]

    # ---------- 5. sklearn-like wrapper ----------
    class _KMeansWrapper:
        def __init__(self, centroids, labels):
            self.cluster_centers_ = centroids
            self.labels_ = labels

    return _KMeansWrapper(kmeans.centroids, labels)


import torch
import numpy as np
from typing import Tuple


def perform_clustering(
    embeddings: torch.Tensor,
    num_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Product Key Clustering (Row + Residual)

    - 不做全量 astype / contiguous
    - residual 分批计算
    """
    assert embeddings.ndim == 2, "embeddings must be [N, dim]"
    device = embeddings.device

    # ---------- Step 0: to numpy（不拷贝） ----------
    data_np = embeddings.detach().cpu().numpy()
    n, d = data_np.shape

    # ---------- Step 1: Row clustering ----------
    print(f"[FAISS] Row clustering (K={num_clusters})...")
    kmeans_row = train_kmeans(
        data_np,
        num_clusters,
        batch_size=50_000,
    )
    row_keys = kmeans_row.cluster_centers_        # [K, d]
    row_labels = kmeans_row.labels_               # [N]

    # ---------- Step 2: Residuals（分批，避免峰值） ----------
    residuals = np.empty_like(data_np, dtype=np.float32)

    for start in range(0, n, 50_000):
        end = min(start + 50_000, n)
        residuals[start:end] = (
            data_np[start:end].astype(np.float32, copy=False)
            - row_keys[row_labels[start:end]]
        )

    # ---------- Step 3: Column clustering ----------
    print(f"[FAISS] Column clustering (K={num_clusters})...")
    kmeans_col = train_kmeans(
        residuals,
        num_clusters,
        batch_size=50_000,
    )
    col_keys = kmeans_col.cluster_centers_        # [K, d]
    col_labels = kmeans_col.labels_               # [N]

    # ---------- Step 4: Grid index ----------
    grid_indices = row_labels * num_clusters + col_labels

    # ---------- Step 5: back to torch ----------
    return (
        torch.from_numpy(row_keys).to(device=device, dtype=torch.float32),
        torch.from_numpy(col_keys).to(device=device, dtype=torch.float32),
        torch.from_numpy(grid_indices).to(device=device, dtype=torch.long),
    )

