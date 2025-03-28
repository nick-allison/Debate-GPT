import torch

@torch.no_grad()
def estimate_loss(dataset, model, eval_iters=50):
    device = next(model.parameters()).device
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = torch.tensor(losses).mean()
    model.train()
    return out

def train_loop(dataset, model, epochs=1000, report=100, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(epochs):
        x, y = dataset.get_batch('train', device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % report == 0 or step == epochs - 1:
            stats = estimate_loss(dataset, model)
            print(f"[Step {step}] train loss={stats['train']:.4f}, val loss={stats['val']:.4f}")
