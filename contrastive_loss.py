import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, initial_batch_size, temperature=0.5, device='cpu'):
        super(ContrastiveLoss, self).__init__()
        self.initial_batch_size = initial_batch_size # Store initial to handle last batch if smaller
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        # Initial mask, will be recomputed if batch size changes
        self.mask = self._make_mask(self.initial_batch_size).to(self.device)


    def _make_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        current_batch_size = z_i.size(0)

        if current_batch_size == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if current_batch_size != self.mask.size(0) // 2:
            self.mask = self._make_mask(current_batch_size).to(self.device)

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, current_batch_size)
        sim_ji = torch.diag(similarity_matrix, -current_batch_size)

        positive_samples = torch.cat([sim_ij, sim_ji], dim=0).reshape(2 * current_batch_size, 1)
        negative_samples = similarity_matrix[self.mask].reshape(2 * current_batch_size, -1)
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * current_batch_size).to(logits.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * current_batch_size)