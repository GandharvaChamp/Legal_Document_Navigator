import json
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


@dataclass
class DataCfg:
    corpus_dir: str
    benchmarks_dir: str
    split: str  # "dev" or "test"


@dataclass
class ChunkingCfg:
    method: str
    chunk_size: int
    overlap: int


@dataclass
class EmbeddingCfg:
    model: str


@dataclass
class RetrievalCfg:
    type: str  # dense | bm25 | hybrid_rrf | hybrid_weighted
    top_k: int


@dataclass
class RerankCfg:
    enabled: bool
    model: str = ""
    top_n: int = 50


@dataclass
class GenerationCfg:
    model: str
    temperature: float


@dataclass
class Config:
    run_name: str
    seed: int
    data: DataCfg
    chunking: ChunkingCfg
    embedding: EmbeddingCfg
    retrieval: RetrievalCfg
    rerank: RerankCfg
    generation: GenerationCfg


def load(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(
        run_name=raw["run_name"],
        seed=raw["seed"],
        data=DataCfg(**raw["data"]),
        chunking=ChunkingCfg(**raw["chunking"]),
        embedding=EmbeddingCfg(**raw["embedding"]),
        retrieval=RetrievalCfg(**raw["retrieval"]),
        rerank=RerankCfg(**raw["rerank"]),
        generation=GenerationCfg(**raw["generation"]),
    )
    assert cfg.data.split in ("dev", "test"), "split must be dev or test"
    assert cfg.retrieval.type in ("dense", "bm25", "hybrid_rrf", "hybrid_weighted")
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # add torch.manual_seed(seed) here once torch is installed


def start_run(cfg: Config, config_path: str) -> Path:
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.run_name}"
    run_dir = Path("experiments/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(config_path, run_dir / "config.yaml")
    with open(run_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    return run_dir