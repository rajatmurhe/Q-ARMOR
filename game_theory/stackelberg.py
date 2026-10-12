from .attacker import AdaptiveAttacker
from .defender import AdaptiveDefender
from .payoff import SecurityPayoff


class StackelbergSecurityGame:
    """
    Repeated Stackelberg-style security game.

    Defender = Leader
    Attacker = Follower

    Process:

        1. Defender evaluates every defence action.
        2. Attacker observes the defence.
        3. Attacker selects the best response.
        4. Defender utility is calculated.
        5. Defender selects the action with
           maximum anticipated utility.
    """

    def __init__(
        self,
        attack_classes,
        severity=None,
        payoff_model=None
    ):
        self.attack_classes = list(
            attack_classes
        )

        if severity is None:
            severity = {
                attack: 1.0
                for attack in attack_classes
            }

        self.severity = severity

        self.attacker = AdaptiveAttacker(
            attack_classes=self.attack_classes,
            severity=self.severity
        )

        self.defender = AdaptiveDefender()

        if payoff_model is None:
            payoff_model = SecurityPayoff()

        self.payoff_model = payoff_model

        self.history = []

    def evaluate_action(
        self,
        defense_action,
        class_f1
    ):
        """
        Evaluate one defender action while anticipating
        the attacker's best response.
        """

        effectiveness = (
            self.defender.defense_effectiveness(
                action=defense_action,
                attack_classes=self.attack_classes,
                class_f1=class_f1
            )
        )

        attacker_response = (
            self.attacker.best_response(
                class_f1=class_f1,
                defense_effectiveness=effectiveness
            )
        )

        selected_attack = (
            attacker_response["attack"]
        )

        attacker_utility = (
            attacker_response["utility"]
        )

        costs = self.defender.get_costs(
            defense_action
        )

        payoff = (
            self.payoff_model.defender_utility(
                attack=selected_attack,

                attack_severity=
                self.severity.get(
                    selected_attack,
                    1.0
                ),

                attacker_utility=
                attacker_utility,

                costs=costs
            )
        )

        return {
            "defense_action":
                defense_action,

            "attacker_best_response":
                selected_attack,

            "attacker_utility":
                attacker_utility,

            "defense_effectiveness":
                effectiveness,

            "attacker_utilities":
                attacker_response["utilities"],

            "defender_payoff":
                payoff
        }

    def solve(
        self,
        class_f1,
        defense_actions=None,
    ):
        """
        Solve the Stackelberg game.

        The defender evaluates all available actions
        and chooses the one with maximum anticipated utility.
        """

        available = self.defender.get_actions()
        actions = available if defense_actions is None else list(defense_actions)
        if (not actions or len(actions) != len(set(actions))
                or any(action not in available for action in actions)):
            raise ValueError("defense_actions must be a nonempty unique subset of available actions")

        action_results = []

        for action in actions:

            result = self.evaluate_action(
                defense_action=action,
                class_f1=class_f1
            )

            action_results.append(
                result
            )

        best_result = max(
            action_results,

            key=lambda item:
            item["defender_payoff"]["utility"]
        )

        equilibrium = {
            "recommended_defense":
                best_result["defense_action"],

            "anticipated_attack":
                best_result[
                    "attacker_best_response"
                ],

            "attacker_utility":
                best_result[
                    "attacker_utility"
                ],

            "defender_utility":
                best_result[
                    "defender_payoff"
                ]["utility"],

            "all_action_results":
                action_results
        }

        self.history.append(
            equilibrium
        )

        return equilibrium

    def get_history(self):
        """
        Return previous repeated-game decisions.
        """

        return list(
            self.history
        )
