"""Identity -- the entity's core identity, loaded from a config file.

Defines who the entity IS when it comes into existence: name, personality,
seed interests, and initial goal. Loaded from a YAML or JSON config file.

Config is optional -- the entity boots fine without one (defaults to current behavior).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class Identity:
    """The entity's core identity."""

    name: str = "Ryn"
    personality: str = ""               # injected into all LLM system prompts
    seed_interests: list[str] = field(default_factory=list)
    seed_goal: str = ""                 # initial purpose/goal
    seed_goal_category: str = "self"    # goal category

    @classmethod
    def default(cls) -> Identity:
        """Return default identity (backward-compatible empty state)."""
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> Identity:
        """Load identity from a YAML or JSON config file.

        Falls back to defaults on any error (malformed file, missing file, etc.).
        The entity always boots.
        """
        path = Path(path)

        if not path.exists():
            logger.warning("Identity config not found: %s -- using defaults", path)
            return cls.default()

        try:
            text = path.read_text(encoding="utf-8")

            if path.suffix in (".yaml", ".yml"):
                if yaml is None:
                    raise ImportError(
                        "PyYAML required for YAML config. "
                        "Install with: pip install pyyaml"
                    )
                data = yaml.safe_load(text)
            elif path.suffix == ".json":
                data = json.loads(text)
            else:
                # Try YAML first, then JSON
                if yaml is not None:
                    data = yaml.safe_load(text)
                else:
                    data = json.loads(text)

            if not isinstance(data, dict):
                logger.warning("Identity config is not a dict: %s -- using defaults", path)
                return cls.default()

            return cls._from_dict(data)

        except Exception as exc:
            logger.warning("Failed to load identity from %s: %s -- using defaults", path, exc)
            return cls.default()

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Identity:
        """Construct Identity from a config dict, using defaults for missing fields."""
        return cls(
            name=str(data.get("name", "Ryn")),
            personality=str(data.get("personality", "")).strip(),
            seed_interests=[str(s) for s in data.get("seed_interests", [])],
            seed_goal=str(data.get("seed_goal", "")),
            seed_goal_category=str(data.get("seed_goal_category", "self")),
        )
