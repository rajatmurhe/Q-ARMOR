"""Severity-weighted protection benefit minus configurable defence costs."""
class SecurityPayoff:
    def __init__(self,security_weight=1.0,false_positive_weight=.50,latency_weight=.30,resource_weight=.30):
        import math
        values=[security_weight,false_positive_weight,latency_weight,resource_weight]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Payoff weights must be finite and nonnegative')
        self.security_weight=security_weight
        self.false_positive_weight=false_positive_weight
        self.latency_weight=latency_weight
        self.resource_weight=resource_weight

    def attacker_utility(self,attack,attack_utility):return float(attack_utility)

    def defender_utility(self,attack,attack_severity,attacker_utility,costs):
        # Attacker utility already includes severity. Applying severity again
        # squared the residual-risk penalty for nonuniform severity values.
        security_benefit=float(attack_severity)-float(attacker_utility)
        false_positive_cost=float(costs['false_positive_cost'])
        latency_cost=float(costs['latency_cost'])
        resource_cost=float(costs['resource_cost'])
        utility=(self.security_weight*security_benefit-self.false_positive_weight*false_positive_cost
                 -self.latency_weight*latency_cost-self.resource_weight*resource_cost)
        return dict(attack=attack,security_benefit=security_benefit,false_positive_cost=false_positive_cost,
                    latency_cost=latency_cost,resource_cost=resource_cost,utility=float(utility))
