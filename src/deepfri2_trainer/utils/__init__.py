"""The deepFRI2 training procedure.

- ``dataloader``: ``DeepFRIDataset`` and the train/eval/test DataLoader factories
- ``training``:   optimizer/loss setup, train & eval loops, metrics, wandb metric logging
- ``losses``:     ``WeightedFocalLoss`` (structure) and ``MCMLossDAG`` (sequence, fusion)
"""
