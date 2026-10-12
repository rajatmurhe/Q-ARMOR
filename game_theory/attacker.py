"""Adaptive attacker: measured weakness times severity, reduced by defence."""
import numpy as np


class AdaptiveAttacker:
    def __init__(self, attack_classes, severity=None):
        self.attack_classes = list(attack_classes)
        self.severity = severity if severity is not None else {a: 1.0 for a in self.attack_classes}

    def calculate_base_risk(self, class_f1):
        return np.array([(1.0-max(0.,min(1.,float(class_f1.get(a,0.))))) *
                         float(self.severity.get(a,1.)) for a in self.attack_classes], dtype=float)

    def best_response(self, class_f1, defense_effectiveness):
        risk = self.calculate_base_risk(class_f1)
        utilities=np.array([risk[i]*(1.-max(0.,min(1.,float(defense_effectiveness.get(a,0.)))))
                            for i,a in enumerate(self.attack_classes)])
        i=int(np.argmax(utilities))
        return {'attack':self.attack_classes[i], 'utility':float(utilities[i]),
                'utilities':{a:float(utilities[j]) for j,a in enumerate(self.attack_classes)}}

    def attack_distribution(self,class_f1,defense_effectiveness,temperature=.20):
        result=self.best_response(class_f1,defense_effectiveness)
        utilities=np.array([result['utilities'][a] for a in self.attack_classes])
        if temperature<=0:temperature=.20
        scaled=utilities/temperature;scaled-=np.max(scaled)
        probabilities=np.exp(scaled);probabilities/=probabilities.sum()
        return {a:float(probabilities[j]) for j,a in enumerate(self.attack_classes)}
