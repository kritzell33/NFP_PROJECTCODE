"""Crew relationship graph.

A weighted undirected graph over crew members. Positive interactions
(socializing, cooperative work) strengthen edges; friction events weaken
them. Team-level cohesion is summarized from edge weights, and the friction
hazard in the event director feeds on low cohesion - closing the social
feedback loop.

Edge weights live in [-1, 1]: 0 is "strangers", 1 is strong positive bond,
negative values are active antagonism.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class RelationshipGraph:
    def __init__(self, agent_ids: list[str]):
        self.g = nx.Graph()
        self.g.add_nodes_from(agent_ids)
        for a, b in nx.non_edges(self.g.copy()):
            self.g.add_edge(a, b, weight=0.0)
        # non_edges on a graph with no edges yields all pairs; the copy()
        # avoids mutating while iterating.

    def adjust(self, a: str, b: str, delta: float) -> None:
        if a == b:
            return
        w = self.g[a][b]["weight"] + delta
        self.g[a][b]["weight"] = float(np.clip(w, -1.0, 1.0))

    def pair_weight(self, a: str, b: str) -> float:
        return float(self.g[a][b]["weight"])

    def decay(self, factor: float = 0.97) -> None:
        """Relationships drift toward neutral without maintenance.

        Applied daily by the mission loop; keeps cohesion from saturating and
        makes sustained bonding an active behavior, matching the isolation
        literature's picture of cohesion as work rather than a set-point.
        """
        for _, _, d in self.g.edges(data=True):
            d["weight"] = float(d["weight"] * factor)

    def cohesion(self) -> float:
        """Mean edge weight across the crew, in [-1, 1]."""
        weights = [d["weight"] for _, _, d in self.g.edges(data=True)]
        return float(np.mean(weights)) if weights else 0.0

    def most_strained_pair(self, rng: np.random.Generator) -> tuple[str, str]:
        """Sample a pair for a friction event, biased toward weak/negative edges."""
        edges = list(self.g.edges(data=True))
        # Convert weights to friction propensity: weaker bond -> higher chance.
        propensity = np.array([1.0 - d["weight"] for _, _, d in edges], dtype=float)
        propensity = np.clip(propensity, 0.05, None)
        probs = propensity / propensity.sum()
        idx = int(rng.choice(len(edges), p=probs))
        a, b, _ = edges[idx]
        return a, b
