"""The deepFRI2 training procedure, and the CAFA evaluation of what it produces.

- ``dataloader``: ``DeepFRIDataset`` and the train/eval/test DataLoader factories
- ``training``:   optimizer/loss setup, train & eval loops, metrics, wandb metric logging
- ``losses``:     ``WeightedFocalLoss`` (structure) and ``MCLossDAG`` (sequence, fusion)
- ``target_matrix``: protein -> GO-term supervision from the annotation tables (``preprocess.py``)
- ``split``:      homology-aware train/eval split (MMseqs2 clustering + balanced assignment)
- ``evaluator``:  CAFA scores for trained runs, deepFRI v1 and the competitors (``validate.ipynb``)
- ``figures``:    figures and tables from those scores
"""
