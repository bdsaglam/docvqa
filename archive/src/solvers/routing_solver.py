"""Routing solver — routes documents to different solvers based on category.

Uses category-based rules to pick the best solver per document.
Each category can be mapped to a different solver configuration,
allowing per-domain optimization.
"""

from __future__ import annotations

import logging
from typing import Any

from docvqa.data import Document

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependency issues
SOLVER_FACTORIES = {
    "flat_solo": "docvqa.solvers.flat_solo_solver:create_flat_solo_program",
    "leanest_solo": "docvqa.solvers.leanest_solo_solver:create_leanest_solo_program",
    "lean_solo": "docvqa.solvers.lean_solo_solver:create_lean_solo_program",
    "flat_batch": "docvqa.solvers.flat_batch_solver:create_flat_batch_program",
}


def _create_solver(solver_type: str, cfg: dict[str, Any]):
    """Lazy-import and create a solver by type name."""
    if solver_type not in SOLVER_FACTORIES:
        raise ValueError(f"Unknown solver type '{solver_type}'. Available: {list(SOLVER_FACTORIES.keys())}")
    module_path, factory_name = SOLVER_FACTORIES[solver_type].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    return factory(**cfg)


class RoutingSolver:
    """Routes documents to category-specific solvers.

    Default routing:
    - engineering_drawing, business_report, comics → flat_solo (visual-heavy)
    - everything else → flat_batch (text-heavy)
    """

    DEFAULT_VISUAL_CATEGORIES = {"engineering_drawing", "business_report", "comics"}

    def __init__(
        self,
        default_solver: Any,
        category_solvers: dict[str, Any] | None = None,
        visual_categories: set[str] | None = None,
        visual_solver: Any | None = None,
    ):
        self.default_solver = default_solver
        self.category_solvers = category_solvers or {}
        self.visual_categories = visual_categories or self.DEFAULT_VISUAL_CATEGORIES
        self.visual_solver = visual_solver

    def _pick_solver(self, document: Document):
        """Pick solver based on document category."""
        # Exact category override first
        if document.doc_category in self.category_solvers:
            solver = self.category_solvers[document.doc_category]
            logger.info("Routing %s (category=%s) -> %s (category override)",
                        document.doc_id, document.doc_category, solver.__class__.__name__)
            return solver

        # Visual categories → visual_solver
        if self.visual_solver and document.doc_category in self.visual_categories:
            logger.info("Routing %s (category=%s) -> %s (visual)",
                        document.doc_id, document.doc_category, self.visual_solver.__class__.__name__)
            return self.visual_solver

        logger.info("Routing %s (category=%s) -> %s (default)",
                    document.doc_id, document.doc_category, self.default_solver.__class__.__name__)
        return self.default_solver

    def solve_document(self, document: Document) -> tuple[dict[str, str], dict[str, list[dict]]]:
        solver = self._pick_solver(document)
        return solver.solve_document(document)


def create_routing_solver(
    default_type: str = "flat_batch",
    default_config: dict[str, Any] | None = None,
    visual_type: str = "flat_solo",
    visual_config: dict[str, Any] | None = None,
    visual_categories: list[str] | None = None,
    category_overrides: dict[str, dict[str, Any]] | None = None,
) -> RoutingSolver:
    """Create a routing solver with configurable per-category solver selection.

    Args:
        default_type: Solver type for unmapped categories.
        default_config: Config dict passed to the default solver factory.
        visual_type: Solver type for visual-heavy categories.
        visual_config: Config dict passed to the visual solver factory.
        visual_categories: Categories to route to the visual solver.
        category_overrides: Per-category solver overrides. Keys are category
            names, values are dicts with 'type' and 'config' keys.
    """
    default_solver = _create_solver(default_type, default_config or {})
    visual_solver = _create_solver(visual_type, visual_config or {})

    cat_solvers = {}
    if category_overrides:
        for cat, cfg in category_overrides.items():
            cat_solvers[cat] = _create_solver(cfg["type"], cfg.get("config", {}))

    return RoutingSolver(
        default_solver=default_solver,
        visual_solver=visual_solver,
        visual_categories=set(visual_categories) if visual_categories else None,
        category_solvers=cat_solvers,
    )
