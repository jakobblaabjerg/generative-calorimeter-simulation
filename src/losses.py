import torch, math

def gmm_loss(pi, mu, log_var, x):

    # negative log likelihood loss
    # x shape: (batch, D)

    device = x.device
    batch_size, K, D = mu.shape    
    x = x.unsqueeze(1).expand(-1, K, -1)  # (batch, K, D)
    
    # gaussian log probability
    log_prob = -0.5 * (log_var + (x - mu)**2 / torch.exp(log_var) + torch.log(torch.tensor(2.0 * math.pi, device=device))).sum(dim=2) # (batch, K)

    # weight by mixture
    weighted_log_prob = torch.log(pi + 1e-8) + log_prob # (batch, K)

    # log-sum-exp trick
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    
    return -log_sum.mean()


LOSSES = {
    "gmm": gmm_loss,
}