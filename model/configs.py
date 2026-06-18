

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    data_root: str = "/data/leecg1219/3DGS_FOR_RECONSTRUCTION_v2"
    anomaly_root: str = "/data/leecg1219/3DGS_Anomaly_recon_v2"
    classes: List[str] = field(default_factory=lambda: [
        "01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale",
        "06Bird", "07Owl", "08Sabertooth", "09Swan", "10Sheep",
        "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot",
        "16Cat", "17Scorpion", "18Obesobeso", "19Bear", "20Puppy"
    ])
    input_dim: int = 14
    n_gaussians: int = 20000
    anomaly_types: List[str] = field(default_factory=lambda: [
        "burrs_recon", "stains_recon", "missing_recon"
    ])


@dataclass
class ModelConfig:
    num_groups: int = 1024
    group_size: int = 32
    embed_dim: int = 384
    encoder_depth: int = 6
    decoder_depth: int = 2
    num_heads: int = 6
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.60
    loss_weight_xyz: float = 1.0
    loss_weight_f_dc: float = 1.0
    loss_weight_opacity: float = 3.0
    loss_weight_scale: float = 2.0
    loss_weight_rotation: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    output_dir: str = "/data/leecg1219/MAE_Recon/output_full_v2"
    save_interval: int = 50
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


@dataclass
class EvalConfig:
    n_iterations: int = 3
    use_topk: bool = True
    topk_ratio: float = 0.05


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    def __repr__(self):
        return (
            f"=== Config ===\n"
            f"[Data] root={self.data.data_root}, classes={len(self.data.classes)}, "
            f"input_dim={self.data.input_dim}, n_gaussians={self.data.n_gaussians}\n"
            f"[Model] groups={self.model.num_groups}, group_size={self.model.group_size}, "
            f"embed_dim={self.model.embed_dim}, mask_ratio={self.model.mask_ratio}\n"
            f"[Train] epochs={self.train.epochs}, batch_size={self.train.batch_size}, "
            f"lr={self.train.lr}\n"
            f"[Eval] n_iter={self.eval.n_iterations}, topk_ratio={self.eval.topk_ratio}\n"
        )


if __name__ == "__main__":
    cfg = Config()
    print(cfg)