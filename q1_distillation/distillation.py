"""distillation.py - the knowledge-distillation loss.

total = alpha * T^2 * KL( softmax(student/T) , softmax(teacher/T) )     <- soft targets with temperature
      + (1 - alpha) * CrossEntropy(student, true label)                  <- hard labels
      + beta * FeatureLoss(student features, teacher features)           <- feature-based distillation
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels, T=4.0, alpha=0.7, beta=2.0, use_feature=True):
        super().__init__()
        self.T, self.alpha, self.beta, self.use_feature = T, alpha, beta, use_feature
        # connector: a 1x1 conv that converts student feature channels into teacher feature channels
        self.connector = (nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
                                        nn.BatchNorm2d(teacher_channels)) if use_feature else None)

    def forward(self, s_logits, t_logits, labels, s_feat=None, t_feat=None):
        s_logits, t_logits = s_logits.float(), t_logits.float()
        hard = F.cross_entropy(s_logits, labels)
        soft = F.kl_div(F.log_softmax(s_logits / self.T, dim=1),
                        F.softmax(t_logits / self.T, dim=1), reduction="batchmean") * (self.T ** 2)
        loss = self.alpha * soft + (1.0 - self.alpha) * hard
        feat = s_logits.new_zeros(())
        if self.use_feature and s_feat is not None:
            s = self.connector(s_feat)
            s = F.adaptive_avg_pool2d(s, t_feat.shape[-2:])        # make the spatial sizes equal
            s = F.normalize(s.float(), dim=1)
            t = F.normalize(t_feat.float(), dim=1)
            feat = ((s - t) ** 2).sum(dim=1).mean()
            loss = loss + self.beta * feat
        return loss, {"hard": hard.detach(), "soft": soft.detach(), "feat": feat.detach()}
