import torch
import torch.nn as nn
import torch.nn.functional as F

class GCODLoss(nn.Module):
    def __init__(self, num_classes, q=0.449, k=3, gamma=1.46, threshold=0.42, label_smoothing=0.1, eps=1e-7):
        """
        Generalized Cross-Entropy with Outlier Detection (GCOD) Loss with Label Smoothing.

        Args:
            num_classes (int): Number of classes.
            q (float): Parameter for the Generalized Cross-Entropy (GCE) part. Defaults to 0.449.
            k (int): Number of smallest confident scores to consider for outlier detection. Defaults to 2.
            gamma (float): Weighting factor for the outlier detection loss. Defaults to 0.6309.
            threshold (float): Confidence score threshold. Defaults to 0.5.
            label_smoothing (float): Label smoothing factor between 0 and 1. Defaults to 0.1.
            eps (float): Small constant for numerical stability. Defaults to 1e-7.
        """
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.k = k
        self.gamma = gamma
        self.threshold = threshold
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(logits.device)
        
        # Apply label smoothing
        smooth_targets = (1 - self.label_smoothing) * targets_one_hot + \
                        self.label_smoothing / self.num_classes

        # Compute softmax with better numerical stability
        logits = logits.clamp(min=-15, max=15)  # Prevent extreme values
        softmax_output = F.softmax(logits, dim=1)
        softmax_output = torch.clamp(softmax_output, min=self.eps, max=1.0)  # Prevent zeros

        # Compute the GCE loss with smoothed targets
        prod = (softmax_output * smooth_targets).sum(dim=1)
        prod = torch.clamp(prod, min=self.eps)  # Prevent zeros
        gce_loss = torch.mean((1. - prod**self.q) / self.q)

        # Compute the outlier detection loss
        confident_scores = 1.0 - softmax_output
        smallest_k_confident_scores, _ = torch.topk(confident_scores, min(self.k, confident_scores.size(1)), 
                                                  dim=1, largest=False)
        noisy_samples = torch.sum(smallest_k_confident_scores, dim=1) < self.threshold
        
        # Handle case when no noisy samples are found
        if torch.any(noisy_samples):
            outlier_loss = torch.mean(smallest_k_confident_scores[noisy_samples])
        else:
            outlier_loss = torch.tensor(0.0).to(logits.device)

        # Combine GCE and outlier detection loss
        total_loss = gce_loss + self.gamma * outlier_loss

        return total_loss