import math


class AdaptiveDefender:
    """
    Stackelberg defender.

    The defender acts as the leader and chooses a security policy
    before the attacker selects its best response.
    """

    def __init__(self):
        self.actions = [
            "NORMAL",
            "STRICT_THRESHOLD",
            "CLASS_REWEIGHTING",
            "TARGETED_DEFENSE"
        ]

        # Relative computational cost of each defence.
        self.resource_cost = {
            "NORMAL": 0.05,
            "STRICT_THRESHOLD": 0.15,
            "CLASS_REWEIGHTING": 0.30,
            "TARGETED_DEFENSE": 0.45
        }

        # Relative latency cost.
        self.latency_cost = {
            "NORMAL": 0.05,
            "STRICT_THRESHOLD": 0.10,
            "CLASS_REWEIGHTING": 0.25,
            "TARGETED_DEFENSE": 0.35
        }

        # Expected false-positive cost.
        self.false_positive_cost = {
            "NORMAL": 0.05,
            "STRICT_THRESHOLD": 0.30,
            "CLASS_REWEIGHTING": 0.15,
            "TARGETED_DEFENSE": 0.10
        }

    def get_actions(self):
        """
        Return available defender actions.
        """

        return list(self.actions)

    def get_costs(self, action):
        """
        Return resource, latency and false-positive costs
        for a defence action.
        """

        if action not in self.actions:
            raise ValueError(
                f"Unknown defence action: {action}"
            )

        return {
            "resource_cost":
                self.resource_cost[action],

            "latency_cost":
                self.latency_cost[action],

            "false_positive_cost":
                self.false_positive_cost[action]
        }

    def defense_effectiveness(
        self,
        action,
        attack_classes,
        class_f1
    ):
        """
        Estimate how effective a selected defence action is
        against each attack class.

        IMPORTANT:
        These initial values are policy-model assumptions for
        prototype testing. Later we will replace/calibrate them
        using measured experimental results from the IDS.
        """

        if action not in self.actions:
            raise ValueError(
                f"Unknown defence action: {action}"
            )

        effectiveness = {}
        parameters = getattr(self, "effectiveness_parameters", {})
        for key, pair in parameters.items():
            if key not in self.actions or len(pair) != 2 or any(
                not math.isfinite(v) or v < 0 for v in pair
            ):
                raise ValueError("Invalid effectiveness parameters")

        for attack in attack_classes:

            f1 = float(
                class_f1.get(attack, 0.0)
            )

            f1 = max(
                0.0,
                min(1.0, f1)
            )

            weakness = 1.0 - f1

            if action == "NORMAL":

                value = 0.10

            elif action == "STRICT_THRESHOLD":

                # Better detection, but potentially more
                # false positives.
                value = 0.25 + (
                    0.20 * weakness
                )

            elif action == "CLASS_REWEIGHTING":

                # More useful for weak classes.
                value = 0.20 + (
                    0.50 * weakness
                )

            elif action == "TARGETED_DEFENSE":

                # Strongest protection for weak attack classes.
                value = 0.30 + (
                    0.60 * weakness
                )

            else:
                value = 0.0

            if action in parameters:
                intercept, slope = parameters[action]
                value = intercept + slope * weakness
            effectiveness[attack] = min(
                0.95,
                max(0.0, value)
            )

        return effectiveness