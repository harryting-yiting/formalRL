import random

from model import Model
from automata import Fsa
from typing import Dict
import re
from random import choice
import sympy as sp

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
        return PredicateEvaluationResult(-self.result)

    def get_result(self):
        return self.result


class Predicate:
    def __init__(self, tmp=0, actionable=False):
        self.actionable: bool = actionable
        self.tmp = tmp
        pass

    def actionable(self) -> bool:
        return self.actionable

    def evaluate(self, s: State) -> PredicateEvaluationResult:
        return PredicateEvaluationResult(self.tmp)


class Fspa(Fsa):

    def __init__(self, name="Fspa", predicates: Dict[str, Predicate] = None, multi=True):
        super().__init__(name, predicates.keys(), multi)
        self.PREDICATE_DICT = predicates  # const

    def compute_guard(self, guard: str, s: State):
        # A & B | C
        # Handle (1)
        guard = re.sub(r'\(1\)', 'PredicateEvaluationResult(1)', guard)
        # Handle (0)
        guard = re.sub(r'\(0\)', 'PredicateEvaluationResult(0)', guard)

        # Convert logic connectives
        guard = re.sub(r'\&\&', '&', guard)
        guard = re.sub(r'\|\|', '|', guard)
        #guard = re.sub(r'~', '!', guard)
        used_pds = []
        for key in self.PREDICATE_DICT.keys():
            guard = re.sub(r'\b{}\b'.format(key),
                           "self.PREDICATE_DICT['{}'].evaluate(s)".format(key),
                           guard)
        print(guard)
        return eval(guard).get_result()

    def guard_from_bitmaps(self, bitmaps: set):
        guard_expr: str = ''
        len_prop = len(self.props)
        for counter, conjunction in enumerate(bitmaps):
            guard_expr +='('
            for counter_d, prop in enumerate(self.props):
                if self.props[prop] & conjunction:
                    guard_expr += prop
                else:
                    guard_expr += '~'
                    guard_expr += prop
                if counter_d < (len_prop - 1):
                    guard_expr += ' & '

            guard_expr += ')'
            if counter < (len(bitmaps) - 1):
                guard_expr += '|'

        print(guard_expr)
        dnf_expr = sp.simplify_logic(guard_expr, 'dnf')
        print(dnf_expr)
        return guard_expr

    def compute_edge_guard(self, edge, s: State):
        return self.compute_guard(edge['guard'], s)

    def compute_node_outgoing(self, q, s: State):
        guards = []
        for _, v, d in self.g.out_edges_iter(q, data=True):
            guards.append(self.compute_edge_guard(d, s))
        # end state
        max_value = max(guards)
        max_index = guards.index(max_value)
        return max_value, max_index

    def next_states_from_mdp_state(self, q, s: State) -> list:
        return [v for _, v, d in self.g.out_edges_iter(q, data=True) if self.compute_edge_guard(d, s)]

    def next_state_from_mdp_state(self, q, s: State):
        nq = self.next_states_from_State(q, s)
        assert len(nq) <= 1
        if nq:
            return nq[0]
        return None

    def get_init_nodes(self) -> list:
        ini_nodes = self.init.keys()
        return [ini_nodes]

    def get_random_non_final_node(self):
        node = choice(self.g.nodes())
        while node in self.final:
            node = choice(self.g.nodes())
        return node


def test_fsa(pds):
    specs = ['F a && F !b']

    for spec in specs:
        aut = Fspa('Fspa', predicates=pds)
        aut.from_formula(spec)
        exp = aut.guard_from_bitmaps({1, 7, 2, 3, 4})
        p = aut.compute_guard(exp, State)
        print(p)



if __name__ == "__main__":

    pds = {'a': Predicate(0.5), 'b': Predicate(2), 'c': Predicate(5)}
    a = PredicateEvaluationResult(5)
    b = PredicateEvaluationResult(6)
    c = PredicateEvaluationResult(1)


    test_fsa(pds)








