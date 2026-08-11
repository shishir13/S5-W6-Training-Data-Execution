"""Shard manifest — schema and validation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class ShardManifest:
    shard_id: str
    lane: str
    shard_file: str        # relative path from artifacts root
    token_count: int
    sha256_content: str    # hex digest of raw token bytes in shard file
    tokenizer_id: str
    tokenizer_hash: str
    created_at: str        # ISO-8601 UTC
    is_eval: bool
    doc_count: int
    source_doc_ids: List[str]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ShardManifest":
        data = json.loads(path.read_text())
        return cls(**data)

    def validate_file(self, artifacts_root: Path) -> bool:
        """Return True if shard file exists and its hash matches manifest."""
        import hashlib
        shard_path = artifacts_root / self.shard_file
        if not shard_path.exists():
            return False
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        return actual_hash == self.sha256_content
