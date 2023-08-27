#! /usr/bin/python
from automata import Fsa
from typing import Dict
import re
from random import choice


class State:
    pass


class PredicateEvaluationResult:
    def __init__(self, result: float):
        self.result = result

    def __and__(self, other):
        return PredicateEvaluationResult(min(self.result, other.result))

    def __or__(self, other):
        return PredicateEvaluationResult(max(self.result, other.result))

    def __invert__(self):
        if self.result == 0:
            return PredicateEvaluationResult(-0.00001)
        return PredicateEvaluationResult(-self.result)

    def get_result(self):
        if self.result == 0:
            return 0.000001
        return self.result


class Predicate:
    def __init__(self, actionable=False):
        self.actionable: bool = actionable
        pass

    def actionable(self) -> bool:
        return self.actionable

    def evaluate(self, s: State) -> PredicateEvaluationResult:
        return PredicateEvaluationResult(0)


class Fspa(Fsa):

    def __init__(self, name="Fspa", predicates: Dict[str, Predicate] = None, multi=True):
        super().__init__(name, list(predicates.keys()), multi)
        self.PREDICATE_DICT = predicates  # const

    def compute_guard(self, guard: str, s: State):
        # A & B | C
        # Handle (1)
        guard = re.sub(r'\(1\)', 'PredicateEvaluationResult(1)', guard)
        guard = re.sub(r'\(true\)', 'PredicateEvaluationResult(1)', guard)
        # Handle (0)
        guard = re.sub(r'\(0\)', 'PredicateEvaluationResult(0)', guard)

        # Convert logic connectives
        guard = re.sub(r'\&\&', '&', guard)
        guard = re.sub(r'\|\|', '|', guard)
        guard = re.sub(r'!', '~', guard)
        used_pds = []
        for key in self.PREDICATE_DICT.keys():
            # The predicate may have internal state, but the best practice is not having
            # TODO: make restrictions on the internal state
            guard = re.sub(r'\b{}\b'.format(key),
                           "self.PREDICATE_DICT['{}'].evaluate(s)".format(key),
                           guard)
        return eval(guard).get_result()

    def update_out_edge_predicates(self, q, s: State):
        """
        compute edge predicate come out of q, store them at "weight'
        """
        for _, v, d in self.g.out_edges(q, data=True):
            d['weight'] = self.compute_guard(d['guard'], s)
        return

    def get_reward(self, q):
        # 1. not self
        # 2. not trap state
        # 3. not unactionable
        # TODO: add actionable properties to predicate
        if q in self.trap:
            return -100

        rewards = []
        for _, v, d in self.g.out_edges(q, data=True):
            if v not in (self.trap | set(self.init.keys())):
                rewards.append(d['weight'])
        return max(rewards)

    def get_next_states_from_mdp_state(self, q) -> list:
        return [v for _, v, d in self.g.out_edges(q, data=True) if d['weight'] > 0]

    def get_next_state_from_mdp_state(self, q):
        nq = self.get_next_states_from_mdp_state(q)
        assert len(nq) <= 1
        if nq:
            return nq[0]
        return None

    def get_init_node(self) -> list:
        return list(self.init.keys())[0]

    def get_random_non_final_node(self):
        node = choice(list(self.g.nodes))
        while node in self.final:
            node = choice(list(self.g.nodes()))
        return node

    def copy_from_fsa(self, fsa: Fsa):
        self.g = fsa.g.copy()
        self.init = dict(fsa.init)
        self.final = set(fsa.final)
        self.final = fsa.final
        return

    def determinize(self):
        fsa = super().determinize()
        fspa = Fspa(self.name, self.PREDICATE_DICT)
        fspa.copy_from_fsa(fsa)
        return fspa

def test_fsa(pds):
    specs = ['F a && F !b']

    for spec in specs:
        aut = Fspa('Fspa', predicates=pds)
        aut.from_formula(spec)
        exp = aut.guard_from_bitmaps({1, 7, 2, 3, 4})
        print(exp)
        p = aut.compute_guard(exp, State)
        print(aut)
        auto2 = aut.determinize()
        print(auto2)
        print(p)
        auto2 = aut
        node = auto2.get_random_non_final_node()
        value = auto2.compute_node_outgoing(node, State)
        print(node)
        print(auto2.g.edges())
        print(aut.g.edges())
        print(value)


if __name__ == "__main__":

    pds = {'a': Predicate(0.5), 'b': Predicate(2), 'c': Predicate(5)}
    a = PredicateEvaluationResult(5)
    b = PredicateEvaluationResult(6)
    c = PredicateEvaluationResult(1)


    test_fsa(pds)








