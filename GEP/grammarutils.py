"""
Created on Jan 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import collections
import itertools

import nltk


def get_pcfg(rules, index=False, mapping=None):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()
    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
            elif index and mapping and token[0] == "'":
                tokens[i] = "'{}'".format(mapping[token.strip("'")])
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))
    grammar_rules.extend(non_terminal_rules)
    return grammar_rules


def read_grammar(filename, index=False, mapping=None, insert=True):
    with open(filename) as f:
        rules = [rule.strip() for rule in f.readlines()]
        if insert:
            rules.insert(0, 'GAMMA -> S [1.0]')
        grammar_rules = get_pcfg(rules, index, mapping)
        grammar = nltk.PCFG.fromstring(grammar_rules)
    return grammar


# Earley parser for prediction
def get_production_prob(selected_edge, grammar):
    # Find the corresponding production rule of the edge, and return its probability
    for production in grammar.productions(lhs=selected_edge.lhs()):
        if production.rhs() == selected_edge.rhs():
            return production.prob()


def find_parent(selected_edge, chart):
    # Find the parent edges that lead to the selected edge
    p_edges = list()
    for p_edge in chart.edges():
        # Important: Note that p_edge.end() is not equal to p_edge.start() + p_edge.dot(),
        # when a node in the edge spans several tokens in the sentence
        if p_edge.end() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
            p_edges.append(p_edge)
    return p_edges


def get_edge_prob(selected_edge, chart, grammar):
    # Compute the probability of the edge by recursion
    prob = get_production_prob(selected_edge, grammar)
    if selected_edge.start() != 0:
        parent_prob = 0
        for parent_edge in find_parent(selected_edge, chart):
            parent_prob += get_edge_prob(parent_edge, chart, grammar)
        prob *= parent_prob
    return prob


def remove_duplicate(tokens):
    return [t[0] for t in itertools.groupby(tokens)]


def earley_predict(grammar, tokens):
    tokens = remove_duplicate(tokens)
    symbols = list()
    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return list()
    end_edges = list()

    for edge in e_chart.edges():
        # print(edge)
        if edge.end() == len(tokens):
            # Only add terminal nodes
            if isinstance(edge.nextsym(), unicode):
                symbols.append(edge.nextsym())
                end_edges.append(edge)

    probs = list()
    for end_edge in end_edges:
        probs.append(get_edge_prob(end_edge, e_chart, grammar))

    # Eliminate duplicate
    symbols_no_duplicate = list()
    probs_no_duplicate = list()
    for s, p in zip(symbols, probs):
        if s not in symbols_no_duplicate:
            symbols_no_duplicate.append(s)
            probs_no_duplicate.append(p)
        else:
            probs_no_duplicate[symbols_no_duplicate.index(s)] += p

    return symbols_no_duplicate, probs_no_duplicate


def grammar_to_dot(grammar, filename):
    and_nodes = list()
    or_nodes = list()
    terminal_nodes = list()
    root_branch_count = 0

    edges = list()
    for production in grammar.productions():
        if production.prob() == 1:
            and_nodes.append(str(production.lhs()))
            for i, child_node in enumerate(production.rhs()):
                edges.append(str(production.lhs()) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(unichr(9312+i)))
        else:
            or_nodes.append(str(production.lhs()))
            if str(production.lhs()) == 'S':
                root_branch_count += 1
                and_nodes.append('S{}'.format(root_branch_count))
                edges.append(str(production.lhs()) + ' -> ' + str('S{}'.format(root_branch_count)) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')
                for i, child_node in enumerate(production.rhs()):
                    edges.append(str('S{}'.format(root_branch_count)) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(unichr(9312+i)))
            else:
                for child_node in production.rhs():
                    edges.append(str(production.lhs()) + ' -> ' + str(child_node) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')

        for child_node in production.rhs():
            if isinstance(child_node, unicode):
                terminal_nodes.append(child_node)

    vertices = list()
    and_nodes = set(and_nodes)
    or_nodes = set(or_nodes)
    terminal_nodes = set(terminal_nodes)

    for and_node in and_nodes:
        vertices.append(and_node + ' [shape=doublecircle, fillcolor=green, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for or_node in or_nodes:
        vertices.append(or_node + ' [shape=circle, fillcolor=yellow, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for terminal in terminal_nodes:
        vertices.append(terminal + ' [shape=box, fillcolor=white, style=filled, ranksep=0.5, nodesep=0.5]\n')

    # edges = set(edges)
    with open(filename, 'w') as f:
        f.write('digraph G {\nordering=out\n')

        for vertex in vertices:
            f.write(vertex)

        for edge in edges:
            print(edge)
            f.write(edge.encode('utf-8'))

        f.write('}')


def main():
    pass


if __name__ == '__main__':
    main()
