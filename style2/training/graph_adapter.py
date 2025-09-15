import time
import torch
from . import graph_backend

@torch.cuda.amp.autocast(enabled=False) # graph loss w fp32
def compute_graph_loss(images: torch.Tensor, real_images: torch.Tensor, sub_batch: int | None, slic_kwargs: dict[str, object]) -> tuple[torch.Tensor, float]:
    """
    images: NCHW w [-1,1], float (fake images)
    real_images: NCHW w [-1,1], float (real images for statistics)
    sub_batch: jeśli nie None, tnij batch na mniejsze kawałki
    slic_kwargs: parametry do backendu (n_segments, compactness, sigma, max_iter)
    Returns: (loss_scalar, elapsed_ms)
    """
    t0 = time.time()
    x = images.float()

    if sub_batch is not None and sub_batch > 0 and x.size(0) > sub_batch:
        chunks = torch.split(x, sub_batch, dim=0)
        losses = []
        for c in chunks:
            l = graph_backend.graph_loss(c, real_images=real_images, **slic_kwargs)
            losses.append(l)
        loss = torch.stack(losses).mean()
    else:
        loss = graph_backend.graph_loss(x, real_images=real_images, **slic_kwargs)

    ms = (time.time() - t0) * 1000.0
    return loss, ms
