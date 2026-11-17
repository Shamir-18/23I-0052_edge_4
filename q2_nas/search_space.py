"""search_space.py - the NAS search space + the automatic architecture generator.

An architecture is a small dictionary (JSON), for example
  {"resolution": 32, "stem_channels": 16,
   "stages": [{"block": "dwsep", "kernel": 5, "channels": 48, "blocks": 2, "activation": "silu"}, ...]}
`build_model(cfg)` turns it into a real PyTorch network.
"""
import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

SPACE = {
    "resolution": [24, 32, 40],                  # input picture size (CIFAR image is resized to this)
    "stem_channels": [16, 24, 32],
    "min_stages": 2, "max_stages": 4,            # variable convolutional depth (number of stages)
    "channels": [16, 24, 32, 48, 64, 96, 128],   # variable number of filters
    "kernel": [3, 5, 7],                         # different kernel sizes
    "block": ["conv", "dwsep"],                  # normal conv block or depthwise-separable block
    "blocks": [1, 2, 3],                         # blocks inside a stage (also variable depth)
    "activation": ["relu", "relu6", "silu", "hardswish", "leaky_relu"],
}


def make_act(name):
    return {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU, "hardswish": nn.Hardswish,
            "leaky_relu": lambda: nn.LeakyReLU(0.1)}[name]()


def random_stage(rng):
    return {"block": rng.choice(SPACE["block"]), "kernel": rng.choice(SPACE["kernel"]),
            "channels": rng.choice(SPACE["channels"]), "blocks": rng.choice(SPACE["blocks"]),
            "activation": rng.choice(SPACE["activation"])}


def random_config(rng):
    n = rng.randint(SPACE["min_stages"], SPACE["max_stages"])
    return {"resolution": rng.choice(SPACE["resolution"]), "stem_channels": rng.choice(SPACE["stem_channels"]),
            "stages": [random_stage(rng) for _ in range(n)]}


def _different(rng, options, current):
    return rng.choice([o for o in options if o != current])


def mutate(cfg, rng):
    """Evolution step: copy the parent and change 1-2 things."""
    child = copy.deepcopy(cfg)
    for _ in range(rng.choice([1, 2])):
        stages = child["stages"]
        ops = ["resolution", "stem", "field", "field", "field"]
        if len(stages) < SPACE["max_stages"]:
            ops.append("add_stage")
        if len(stages) > SPACE["min_stages"]:
            ops.append("remove_stage")
        op = rng.choice(ops)
        if op == "resolution":
            child["resolution"] = _different(rng, SPACE["resolution"], child["resolution"])
        elif op == "stem":
            child["stem_channels"] = _different(rng, SPACE["stem_channels"], child["stem_channels"])
        elif op == "field":
            st = rng.choice(stages)
            f = rng.choice(["block", "kernel", "channels", "blocks", "activation"])
            st[f] = _different(rng, SPACE[f], st[f])
        elif op == "add_stage":
            stages.insert(rng.randint(0, len(stages)), random_stage(rng))
        elif op == "remove_stage":
            stages.pop(rng.randrange(len(stages)))
    return child


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True)


class Block(nn.Module):
    """'conv'  : Conv(k x k) -> BN -> activation
       'dwsep' : depthwise Conv(k x k) -> BN -> act -> pointwise Conv(1x1) -> BN -> act   (much cheaper)"""

    def __init__(self, cin, cout, k, stride, kind, act):
        super().__init__()
        p = k // 2
        if kind == "conv":
            layers = [nn.Conv2d(cin, cout, k, stride, p, bias=False), nn.BatchNorm2d(cout), make_act(act)]
        else:
            layers = [nn.Conv2d(cin, cin, k, stride, p, groups=cin, bias=False), nn.BatchNorm2d(cin), make_act(act),
                      nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), make_act(act)]
        self.body = nn.Sequential(*layers)
        self.use_res = (stride == 1 and cin == cout)       # skip connection when shapes match

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_res else y


class SearchNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        stages = cfg["stages"]
        stem = cfg["stem_channels"]
        layers = [nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), make_act(stages[0]["activation"])]
        cin = stem
        for i, st in enumerate(stages):
            for b in range(st["blocks"]):
                stride = 2 if (b == 0 and i > 0) else 1      # every stage after the first halves the picture size
                layers.append(Block(cin, st["channels"], st["kernel"], stride, st["block"], st["activation"]))
                cin = st["channels"]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(cin, num_classes))

    def forward(self, x, return_feat=False):
        f = self.features(x)
        logits = self.head(f)
        return (logits, f) if return_feat else logits


def build_model(cfg, num_classes=10):
    """The automatic architecture generator: configuration (dict) -> network."""
    return SearchNet(cfg, num_classes)
