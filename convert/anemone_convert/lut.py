import torch

KMEANS_STEPS = 300
ROW_BLOCK = 4096
GPTQ_BLOCK = 128
GPTQ_DAMP = 0.01


def boundaries(centroids: torch.Tensor) -> torch.Tensor:
    return ((centroids[:, 1:] + centroids[:, :-1]) / 2).contiguous()


def farthest_first(grid: torch.Tensor, present: torch.Tensor, clusters: int) -> torch.Tensor:
    # Apple's k-means seeds farthest-first, so outlier weights keep their own centroid; quantile seeds wipe them
    low = torch.where(present, grid, torch.inf).amin(dim=1, keepdim=True)
    high = torch.where(present, grid, -torch.inf).amax(dim=1, keepdim=True)
    centroids = torch.cat([low, high], dim=1)
    while centroids.shape[1] < clusters:
        gap = (grid[None, :, None] - centroids[:, None, :]).abs().amin(dim=2)
        far = torch.where(present, gap, -1).argmax(dim=1, keepdim=True)
        centroids = torch.cat([centroids, grid[far]], dim=1)
    return centroids.sort(dim=1).values


def kmeans_1d(values: torch.Tensor, weights: torch.Tensor, clusters: int) -> torch.Tensor:
    """Per-row weighted 1-D k-means; returns sorted centroids of shape (rows, clusters)."""
    grid, inverse = torch.unique(values, return_inverse=True)
    mass = torch.zeros(values.shape[0], grid.numel()).scatter_add_(1, inverse, weights)
    present, spread = mass > 0, grid.expand(values.shape[0], -1).contiguous()
    centroids = farthest_first(grid, present, clusters)
    for _ in range(KMEANS_STEPS):
        segment = torch.searchsorted(boundaries(centroids), spread)
        weight = torch.zeros_like(centroids).scatter_add_(1, segment, mass)
        total = torch.zeros_like(centroids).scatter_add_(1, segment, mass * spread)
        updated = torch.where(weight > 0, total / weight.clamp_min(1e-30), centroids).sort(dim=1).values
        if torch.equal(updated, centroids):
            break
        centroids = updated
    return centroids


def inverse_cholesky(hessian: torch.Tensor) -> torch.Tensor:
    h = hessian.clone()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    h.diagonal().add_(GPTQ_DAMP * h.diagonal().mean())
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)


def gptq_snap(weight: torch.Tensor, luts: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """GPTQ with a fixed per-row LUT: snap one column, push its error onto the columns not yet snapped."""
    w, q = weight.clone(), torch.empty_like(weight)
    bounds, cols = boundaries(luts), weight.shape[1]
    for start in range(0, cols, GPTQ_BLOCK):
        end = min(start + GPTQ_BLOCK, cols)
        errors = torch.empty(w.shape[0], end - start)
        for j in range(start, end):
            q[:, j] = luts.gather(1, torch.searchsorted(bounds, w[:, j : j + 1].contiguous()))[:, 0]
            errors[:, j - start] = (w[:, j] - q[:, j]) / upper[j, j]
            w[:, j + 1 : end] -= errors[:, j - start, None] * upper[j, j + 1 : end]
        w[:, end:] -= errors @ upper[start:end, end:]
    return q


def palettize(weight: torch.Tensor, hessian: torch.Tensor, bits: int = 4, group_rows: int = 32) -> torch.Tensor:
    """Weights with at most 2**bits values per group of `group_rows` output rows, chosen for low output error on the calibration inputs."""
    order = torch.argsort(hessian.diagonal(), descending=True)
    upper = inverse_cholesky(hessian[order][:, order])
    blocks = []
    for start in range(0, weight.shape[0], ROW_BLOCK):
        rows = weight[start : start + ROW_BLOCK]
        groups = rows.reshape(rows.shape[0] // group_rows, group_rows * rows.shape[1])
        # columns count by the energy of their inputs (activation-aware k-means)
        importance = hessian.diagonal().repeat(group_rows).expand_as(groups).contiguous()
        luts = kmeans_1d(groups, importance, 2**bits).repeat_interleave(group_rows, dim=0)
        blocks.append(gptq_snap(rows[:, order], luts, upper)[:, torch.argsort(order)])
    return torch.cat(blocks)


def demo() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    inputs = torch.randn(1024, 256) @ torch.randn(256, 256)
    hessian = inputs.T @ inputs
    palettized = palettize(weight, hessian, bits=4, group_rows=8)
    assert all(palettized[i : i + 8].unique().numel() <= 16 for i in range(0, 64, 8))
    error = ((palettized - weight) @ inputs.T).norm() / (weight @ inputs.T).norm()
    assert error < 0.08, float(error)
    print(f"lut ok, output error {float(error):.4f}")


if __name__ == "__main__":
    demo()
