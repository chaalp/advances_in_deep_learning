from pathlib import Path
import torch

from .low_precision import BigNet4Bit  # reuse 4-bit model

def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    #return None
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net