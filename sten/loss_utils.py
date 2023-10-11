import torch

class PathDisentanglingLoss(torch.nn.Module):
    """Feature Mean-Squared Error. Path disentangling loss
    """

    def __init__(self):
        super(PathDisentanglingLoss, self).__init__()

    def forward(self, p_buffer, ref):
        """Evaluate the metric.
        Args:
            p_buffer(torch.Tensor): (patches, n_samples, embedding_dimensions)
            ref(torch.Tensor):  (patches, n_samples, ref_dimensions)
        """

        if not torch.isfinite(p_buffer).all():
            raise RuntimeError("Infinite loss at train time.")
        if not torch.isfinite(ref).all():
            raise RuntimeError("Infinite loss at train time.")


        loss_p = self.intra_patch_dist(p_buffer, ref)

        loss_b = self.intra_batch_dist(p_buffer, ref)

        return loss_p + loss_b

    def intra_patch_dist(self, p_buffer, ref):
        idx = torch.randperm(p_buffer.shape[1])

        ref_1 = ref
        ref_2 = ref[:, idx, :]
        mse_ref = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=2)

        p_1 = p_buffer
        p_2 = p_buffer[:, idx, :]
        mse_p = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=2)

        loss = 0.5 * torch.mean(torch.pow(mse_ref - mse_p, 2))
        return loss

    def intra_batch_dist(self, p_buffer, ref):
        idx = torch.randperm(p_buffer.shape[0] * p_buffer.shape[1])

        ref_1 = ref.reshape(-1, ref.shape[2])
        ref_2 = ref_1[idx, :]
        mse_ref = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=2)

        p_1 = p_buffer.reshape(-1, p_buffer.shape[2])
        p_2 = p_1[:, idx, :]
        mse_p = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=2)

        loss = 0.5 * torch.mean(torch.pow(mse_ref - mse_p, 2))
        return loss
