"""
Created on Jan 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time

import nltk
import numpy as np

#import config
#import datasets
import grammarutils
import generalizedearley


def parsing_examples():
    rules = list()
    rules.append("GAMMA -> R [1.0]")
    rules.append("R -> R '+' R [0.4]")
    rules.append("R -> '0' [0.3]")
    rules.append("R -> '1' [0.3]")
    grammar_rules = grammarutils.get_pcfg(rules)
    grammar = nltk.PCFG.fromstring(grammar_rules)

    sentence = '0 + 1'
    tokens = sentence.split(' ')

    # earley_parser = nltk.EarleyChartParser(grammar, trace=1)
    # e_chart = earley_parser.chart_parse(tokens)

    symbols = ['0', '1', '+']
    symbol_index = dict()
    for s in symbols:
        symbol_index[s] = symbols.index(s)
    grammar_rules = grammarutils.get_pcfg(rules, index=True, mapping=symbol_index)
    print(rules)
    print(grammar_rules)
    grammar = nltk.PCFG.fromstring(grammar_rules)

    classifier_output = [[0.8, 0.1, 0.1],
                         [0.8, 0.1, 0.1], 
                         [0.1, 0.1, 0.8], 
                         [0.1, 0.8, 0.1], 
                         [0.1, 0.8, 0.1]]
    classifier_output = np.array(classifier_output)
    gen_earley_parser = generalizedearley.GeneralizedEarley(grammar)
    best_string, prob = gen_earley_parser.parse(classifier_output)
    best_string = ' '.join([symbols[int(x)] for x in best_string.split(' ')])
    print(best_string, np.exp(prob))


def test_generalized_earley(grammar, classifier_output):
    gen_earley_parser = generalizedearley.GeneralizedEarley(grammar)
    best_string, prob = gen_earley_parser.parse(classifier_output)
    print('best_string with prob {:.3f}:'.format(prob), best_string)
    print(gen_earley_parser.compute_labels())
    print(np.argmax(classifier_output, axis=1))


def test_earley(grammar, tokens):
    earley_parser = nltk.EarleyChartParser(grammar, trace=1)
    e_chart = earley_parser.chart_parse(tokens)
    for edge in e_chart.edges():
        print(edge, edge.end())

    print(grammarutils.earley_predict(grammar, tokens))


def test_valid():
    paths = config.Paths()
    grammar_file = os.path.join(paths.tmp_root, 'grammar', 'cad', 'stacking_objects.pcfg')

    # sentence = 'null reaching moving placing'
    # grammar = grammarutils.read_grammar(grammar_file, index=False)
    # test_earley(grammar, sentence.split())

    sentence = 'null reaching'
    tokens = sentence.split()
    grammar = grammarutils.read_grammar(grammar_file, index=True, mapping=datasets.cad_metadata.subactivity_index)
    seg_length = 15
    correct_prob = 0.8
    classifier_output = np.ones((seg_length*2, 10)) * 1e-10
    classifier_output[:seg_length, datasets.cad_metadata.subactivity_index[tokens[0]]] = correct_prob
    classifier_output[seg_length:, datasets.cad_metadata.subactivity_index[tokens[1]]] = correct_prob

    classifier_output[:seg_length, datasets.cad_metadata.subactivity_index[tokens[0]]+1] = 1 - correct_prob
    classifier_output[seg_length:, datasets.cad_metadata.subactivity_index[tokens[1]]+1] = 1 - correct_prob
    test_generalized_earley(grammar, classifier_output)


def test_time():
    paths = config.Paths()
    start_time = time.time()
    np.random.seed(int(start_time))
    classifier_output = np.random.rand(100000, 10)
    classifier_output = classifier_output / np.sum(classifier_output, axis=1)[:, None]  # Normalize to probability
    for pcfg in os.listdir(os.path.join(paths.tmp_root, 'grammar', 'cad')):
        if not pcfg.endswith('.pcfg'):
            continue
        grammar_file = os.path.join(paths.tmp_root, 'grammar', 'cad', pcfg)
        grammar = grammarutils.read_grammar(grammar_file, index=True, mapping=datasets.cad_metadata.subactivity_index)
        test_generalized_earley(grammar, classifier_output)
    print('Time elapsed: {}s'.format(time.time() - start_time))


def test_grammar():
    paths = config.Paths()
    for pcfg in os.listdir(os.path.join(paths.tmp_root, 'grammar', 'cad')):
        if not pcfg.endswith('.pcfg'):
            continue
        grammar_file = os.path.join(paths.tmp_root, 'grammar', 'cad', pcfg)
        grammar = grammarutils.read_grammar(grammar_file, index=True, mapping=datasets.cad_metadata.subactivity_index)
        corpus_file = os.path.join(paths.tmp_root, 'corpus', 'cad', pcfg.replace('pcfg', 'txt'))
        with open(corpus_file, 'r') as f:
            for line in f:
                tokens = [str(datasets.cad_metadata.subactivity_index[token]) for token in line.strip(' *#\n').split(' ')]
                earley_parser = nltk.EarleyChartParser(grammar, trace=0)
                e_chart = earley_parser.chart_parse(tokens)
                print(e_chart.edges()[-1])


def visualize_grammar():
    paths = config.Paths()
    dataset_name = 'wnp'
    for pcfg in os.listdir(os.path.join(paths.tmp_root, 'grammar', dataset_name)):
        if not pcfg.endswith('.pcfg'):
            continue
        grammar_file = os.path.join(paths.tmp_root, 'grammar', dataset_name, pcfg)
        grammar = grammarutils.read_grammar(grammar_file, insert=False)
        dot_filename = os.path.join(paths.tmp_root, 'visualize', 'grammar', dataset_name, pcfg.replace('.pcfg', '.dot'))
        pdf_filename = os.path.join(paths.tmp_root, 'visualize', 'grammar', dataset_name, pcfg.replace('.pcfg', '.pdf'))
        grammarutils.grammar_to_dot(grammar, dot_filename)
        os.system('dot -Tpdf {} -o {}'.format(dot_filename, pdf_filename))


def main():
    # test_grammar()
    # test_valid()
    # test_time()
    # visualize_grammar()
    parsing_examples()


if __name__ == '__main__':
    main()
