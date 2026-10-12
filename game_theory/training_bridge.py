"""
Q-ARMOR Stackelberg <-> IDS training bridge.

Purpose
-------
Connect measured validation F1 values from the real IDS training loop to the
existing StackelbergSecurityGame, while keeping the first actuated defence
restricted to CLASS_REWEIGHTING.

Design
------
1. The IDS produces actual per-class F1 on the validation set.
2. All non-benign attack classes are passed directly to Stackelberg.
3. Stackelberg evaluates all four defender actions and the attacker's best
   response for each action.
4. A co-evolutionary probability update is computed from the same utility
   matrices (defender x attack), rather than from class counts.
5. The pure Stackelberg recommendation is actuated only when it is
   CLASS_REWEIGHTING. Other actions are logged but not actuated in this first
   implementation.
6. The selected attack class gets an increased class-weight scale, bounded by
   a maximum to avoid runaway weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .stackelberg import StackelbergSecurityGame


BENIGN_LABELS = {"BenignTraffic", "Benign", "BENIGN"}


@dataclass
class ControllerDecision:
    recommended_defense: str
    anticipated_attack: str
    defender_utility: float
    attacker_utility: float
    class_weight_scale: float
    targeted_class: str | None
    current_class_weights: Dict[int, float]
    p_def: np.ndarray
    p_att: np.ndarray
    U_def: np.ndarray
    U_att: np.ndarray
    class_f1: Dict[str, float]
    game_f1: Dict[str, float]
    all_action_results: list
    applied_action: str | None
    deterioration: bool
    scale_change_reason: str


class StackelbergIDSController:
    """Stateful controller for the repeated IDS/game loop."""

    DEFENDER_ACTIONS = [
        "NORMAL",
        "STRICT_THRESHOLD",
        "CLASS_REWEIGHTING",
        "TARGETED_DEFENSE",
    ]

    def __init__(
        self,
        class_names: Sequence[str],
        base_class_weights: Mapping[int, float] | None = None,
        severity: Mapping[str, float] | None = None,
        scale_init: float = 1.0,
        scale_step: float = 0.15,
        max_scale: float = 2.50,
        min_prob: float = 1e-3,
        replicator_lr: float = 0.50,
        benign_labels: Iterable[str] = BENIGN_LABELS,
        deterioration_tolerance: float = 0.01,
        game_parameters: Mapping | None = None,
        available_defenses: Sequence[str] | None = None,
    ) -> None:
        self.class_names = list(class_names)
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Class names must be unique and ordered by label ID.")
        if not (1 <= scale_init <= max_scale <= 1e12 and 0 <= scale_step <= 1e12
                and 0 <= replicator_lr <= 1 and 0 <= min_prob < 1 / max(4, len(class_names))
                and deterioration_tolerance >= 0):
            raise ValueError("Invalid controller bounds or probability settings.")
        if not np.all(np.isfinite([scale_init, max_scale, scale_step, replicator_lr,
                                  min_prob, deterioration_tolerance])):
            raise ValueError("Controller parameters must be finite.")
        self.deterioration_tolerance = deterioration_tolerance
        self.pending_update = None
        self.scale_change_reason = "none"
        self.deterioration = False
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.benign_labels = set(benign_labels)

        self.attack_classes = [
            name for name in self.class_names if name not in self.benign_labels
        ]
        if not self.attack_classes:
            raise ValueError("No attack classes were found after removing benign labels.")

        # The current game supports arbitrary attack-class names.  Uniform severity
        # is used as the prototype default and can be replaced/calibrated later.
        if severity is None:
            severity = {name: 1.0 for name in self.attack_classes}
        else:
            severity = {name: float(severity.get(name, 1.0)) for name in self.attack_classes}

        if any(not np.isfinite(v) or v < 0 for v in severity.values()):
            raise ValueError("Severity must be finite and nonnegative")
        self.game = StackelbergSecurityGame(
            attack_classes=self.attack_classes,
            severity=severity,
        )

        if game_parameters:
            from .payoff import SecurityPayoff
            if "payoff" in game_parameters:
                self.game.payoff_model = SecurityPayoff(**game_parameters["payoff"])
            for kind in ("resource_cost", "latency_cost", "false_positive_cost"):
                for action, value in game_parameters.get(kind, {}).items():
                    if action not in self.game.defender.get_actions() or not np.isfinite(value) or value < 0:
                        raise ValueError("Invalid game cost override")
                    getattr(self.game.defender, kind)[action] = float(value)
            effects = game_parameters.get("effectiveness", {})
            self.game.defender.effectiveness_parameters = effects
        all_defenses = self.game.defender.get_actions()
        self.DEFENDER_ACTIONS = (all_defenses if available_defenses is None
                                 else list(available_defenses))
        if (not self.DEFENDER_ACTIONS
                or len(self.DEFENDER_ACTIONS) != len(set(self.DEFENDER_ACTIONS))
                or any(action not in all_defenses for action in self.DEFENDER_ACTIONS)):
            raise ValueError("available_defenses must be a nonempty unique subset of game actions")
        self.p_def = np.ones(len(self.DEFENDER_ACTIONS), dtype=float)
        self.p_def /= self.p_def.sum()

        self.p_att = np.ones(len(self.attack_classes), dtype=float)
        self.p_att /= self.p_att.sum()

        self.min_prob = float(min_prob)
        self.replicator_lr = float(replicator_lr)

        self.scale_step = float(scale_step)
        self.max_scale = float(max_scale)
        self.class_scales = {
            class_id: float(scale_init)
            for class_id in range(len(self.class_names))
        }

        if base_class_weights is None:
            base_class_weights = {i: 1.0 for i in range(len(self.class_names))}
        self.base_class_weights = {
            int(k): float(v) for k, v in base_class_weights.items()
        }

        expected = set(range(len(self.class_names)))
        if set(self.base_class_weights) != expected or any(
            not np.isfinite(v) or not 1e-12 <= v <= 1e12 for v in self.base_class_weights.values()
        ):
            raise ValueError("Base weights must contain exactly all label IDs with positive finite values.")
        self.history: list[dict] = []

    @staticmethod
    def _replicator_update(
        probs: np.ndarray,
        utilities: np.ndarray,
        learning_rate: float = 0.50,
        min_prob: float = 1e-3,
    ) -> np.ndarray:
        """Stable discrete replicator update with positive-shift protection."""
        p = np.asarray(probs, dtype=float)
        u = np.asarray(utilities, dtype=float)

        if p.ndim != 1 or u.ndim != 1 or len(p) != len(u):
            raise ValueError("Probability and utility vectors must be 1-D and equal length.")

        if not np.all(np.isfinite(u)):
            raise ValueError("Utilities contain non-finite values.")

        if (not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1)
                or not 0 <= learning_rate <= 1 or not 0 <= min_prob < 1 / len(p)):
            raise ValueError("Invalid probabilities or replicator parameters.")
        # Exponentiated replicator preserves relative utilities and is invariant
        # to additive shifts. Equal utilities correctly leave a uniform state fixed.
        logits = learning_rate * (u - np.max(u))
        new = p * np.exp(np.clip(logits, -700, 0))
        new /= new.sum()
        # Mixture exploration gives an actual lower bound after normalization.
        return (1 - len(p) * min_prob) * new + min_prob

    def _utility_matrices(self, class_f1: Mapping[str, float]):
        """Construct defender/attacker utility matrices for co-evolution."""
        # Use the exact evaluator used by solve(), avoiding two payoff definitions.
        evaluations = [self.game.evaluate_action(action, class_f1)
                       for action in self.DEFENDER_ACTIONS]
        rows_att = [[r["attacker_utilities"][a] for a in self.attack_classes]
                    for r in evaluations]
        rows_def = [[self.game.payoff_model.defender_utility(
            attack=a, attack_severity=self.game.severity[a],
            attacker_utility=r["attacker_utilities"][a],
            costs=self.game.defender.get_costs(r["defense_action"]))["utility"]
                     for a in self.attack_classes] for r in evaluations]
        return np.asarray(rows_def), np.asarray(rows_att)

    def _update_class_scale(self, target_attack: str, class_f1: Mapping[str, float]) -> float:
        class_id = self.class_to_id[target_attack]
        current = float(self.class_scales[class_id])
        weakness = 1.0 - float(np.clip(class_f1.get(target_attack, 0.0), 0.0, 1.0))
        new_scale = min(self.max_scale, current + self.scale_step * weakness)
        if new_scale > current:
            self.pending_update = (class_id, current, dict(class_f1))
            self.scale_change_reason = "increase_for_selected_weak_class"
        else:
            self.scale_change_reason = "scale_unchanged_at_bound_or_zero_weakness"
        self.class_scales[class_id] = new_scale
        return new_scale

    def current_class_weights(self) -> Dict[int, float]:
        """Return base weights multiplied by each class's bounded adaptive scale."""
        raw = np.array([self.base_class_weights[i] * self.class_scales[i]
                        for i in range(len(self.class_names))])
        # Mean-one normalization keeps the overall loss scale comparable to baseline.
        raw /= raw.mean()
        return {i: float(v) for i, v in enumerate(raw)}

    def step(self, class_f1: Mapping[str, float]) -> ControllerDecision:
        """Consume actual measured F1, solve Stackelberg, update co-evolution, actuate reweighting."""
        missing = set(self.attack_classes) - set(class_f1)
        if missing or any(not np.isfinite(v) or not 0 <= v <= 1 for v in class_f1.values()):
            raise ValueError(f"F1 must be finite in [0,1] for every attack class; missing {missing}")
        clean_f1 = {attack: float(class_f1[attack]) for attack in self.attack_classes}
        self.deterioration = False
        self.scale_change_reason = "none"
        if self.pending_update:
            cid, old_scale, previous = self.pending_update
            target = self.class_names[cid]
            drop = (clean_f1[target] < previous[target] - self.deterioration_tolerance
                    or np.mean(list(class_f1.values())) <
                    np.mean(list(previous.values())) - self.deterioration_tolerance)
            if drop:
                self.class_scales[cid] = old_scale
                self.deterioration = True
                self.scale_change_reason = "rollback_after_validation_deterioration"
            self.pending_update = None

        # 1) Actual IDS F1 -> Stackelberg.
        result = self.game.solve(class_f1=clean_f1,
                                 defense_actions=self.DEFENDER_ACTIONS)

        # 2) Utility matrices for the same current state.
        U_def_matrix, U_att_matrix = self._utility_matrices(clean_f1)

        # 3) Co-evolutionary expected utilities.
        U_def = U_def_matrix @ self.p_att
        U_att = U_att_matrix.T @ self.p_def

        self.p_def = self._replicator_update(
            self.p_def,
            U_def,
            learning_rate=self.replicator_lr,
            min_prob=self.min_prob,
        )
        self.p_att = self._replicator_update(
            self.p_att,
            U_att,
            learning_rate=self.replicator_lr,
            min_prob=self.min_prob,
        )

        # 4) Actuate ONLY CLASS_REWEIGHTING initially.
        selected_defense = result["recommended_defense"]
        targeted_class = None
        selected_scale = max(self.class_scales.values(), default=1.0)

        if selected_defense == "CLASS_REWEIGHTING" and not self.deterioration:
            targeted_class = result["anticipated_attack"]
            selected_scale = self._update_class_scale(targeted_class, class_f1)

        decision = ControllerDecision(
            recommended_defense=selected_defense,
            anticipated_attack=result["anticipated_attack"],
            defender_utility=float(result["defender_utility"]),
            attacker_utility=float(result["attacker_utility"]),
            class_weight_scale=float(selected_scale),
            targeted_class=targeted_class,
            current_class_weights=self.current_class_weights(),
            p_def=self.p_def.copy(),
            p_att=self.p_att.copy(),
            U_def=U_def.copy(),
            U_att=U_att.copy(),
            class_f1={k: float(v) for k, v in class_f1.items()},
            game_f1=dict(clean_f1),
            all_action_results=result["all_action_results"],
            applied_action=("CLASS_REWEIGHTING" if targeted_class is not None and self.scale_change_reason == "increase_for_selected_weak_class" else None),
            deterioration=self.deterioration,
            scale_change_reason=self.scale_change_reason,
        )

        self.history.append({
            "recommended_defense": decision.recommended_defense,
            "applied_action": decision.applied_action,
            "deterioration": decision.deterioration,
            "scale_change_reason": decision.scale_change_reason,
            "all_action_results": decision.all_action_results,
            "anticipated_attack": decision.anticipated_attack,
            "defender_utility": decision.defender_utility,
            "attacker_utility": decision.attacker_utility,
            "class_weight_scale": decision.class_weight_scale,
            "targeted_class": decision.targeted_class,
            "p_def": decision.p_def.tolist(),
            "p_att": decision.p_att.tolist(),
            "U_def": decision.U_def.tolist(),
            "U_att": decision.U_att.tolist(),
            "class_f1": decision.class_f1,
            "class_weights": decision.current_class_weights,
        })

        return decision
