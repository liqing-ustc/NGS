from utils import *
import nltk
from GEP import *

rules = [
    "GAMMA -> expression",
    "expression -> term | expression '+' term | expression '-' term",
    "term -> factor | term '*' factor | term '/' factor"
]
num_rule = "factor -> " + ' | '.join(["'%s'"%str(x) for x in range(1, 10)])
rules.append(num_rule)

symbol_index = {x:sym2id(x) for x in sym_list}
grammar_rules = grammarutils.get_pcfg(rules, index=True, mapping=symbol_index)
print('\n'.join(rules))
# print(grammar_rules)
grammar = nltk.CFG.fromstring(grammar_rules)
parser = GeneralizedEarley(grammar)

if __name__ == '__main__':
    classifier_output = [[4.6049e-10, 2.7333e-09, 5.9648e-05, 2.4868e-02, 1.5747e-08, 2.0220e-06,
         3.9989e-07, 2.9596e-04, 5.1052e-07, 9.7476e-01, 1.6015e-05, 1.3034e-07,
         3.3310e-10, 1.7458e-07, 2.6158e-09],
        [1.2057e-09, 4.5331e-11, 2.7309e-01, 1.1915e-09, 2.7949e-06, 7.2661e-01,
         1.9106e-06, 1.0127e-09, 2.5734e-04, 5.3273e-08, 3.2696e-05, 1.6992e-11,
         4.1898e-10, 1.7338e-10, 1.1434e-09],
        [6.7754e-15, 2.1393e-08, 1.7014e-09, 1.8701e-08, 7.8242e-07, 3.9881e-09,
         2.9158e-09, 2.1262e-12, 1.1271e-12, 1.2788e-06, 1.0000e+00, 8.6460e-14,
         2.4007e-13, 3.5565e-13, 9.4279e-14],
        [5.6125e-08, 3.4027e-09, 9.0083e-01, 4.2778e-07, 2.5833e-02, 3.1780e-02,
         3.8263e-02, 1.1287e-08, 3.2831e-03, 1.1298e-11, 2.4219e-08, 1.6487e-06,
         4.2944e-12, 2.5267e-12, 4.8201e-06],
        [3.5127e-09, 2.5219e-08, 3.6138e-05, 4.8666e-01, 5.9548e-06, 1.2554e-05,
         3.2356e-04, 4.7148e-01, 7.7920e-06, 4.1386e-02, 2.2500e-05, 4.4926e-05,
         7.1718e-11, 1.9388e-05, 3.0777e-07],
        [9.6303e-08, 1.7249e-05, 1.1281e-01, 2.2645e-03, 5.7288e-06, 8.6524e-01,
         1.4672e-05, 1.5958e-05, 1.1446e-02, 7.2038e-05, 8.0769e-03, 9.3893e-06,
         2.3787e-05, 9.5070e-06, 1.0722e-06],
        [3.3505e-07, 2.3621e-08, 9.4874e-01, 1.3996e-04, 4.2788e-03, 2.6633e-03,
         1.3446e-05, 3.8993e-06, 3.9327e-02, 8.3370e-05, 4.7459e-03, 2.5074e-07,
         4.3001e-08, 1.5849e-06, 1.8086e-06],
        [4.7227e-10, 8.4414e-10, 1.4390e-03, 3.6918e-06, 6.5836e-01, 2.2979e-02,
         1.6322e-01, 2.6119e-02, 1.2789e-01, 1.1968e-07, 1.4336e-07, 1.0390e-09,
         8.8995e-09, 2.8542e-10, 7.5476e-07],
        [2.6510e-09, 9.5489e-13, 2.1066e-07, 2.6863e-02, 1.5238e-12, 1.4848e-03,
         4.0954e-05, 3.8185e-04, 2.2173e-06, 9.7122e-01, 5.7494e-06, 7.5778e-12,
         1.4150e-10, 6.0840e-13, 1.7209e-10]]
    
    classifier_output = np.array(classifier_output)
    from time import time
    st = time()
    best_string, prob = parser.parse(classifier_output)
    print(time() - st)
    print(best_string)
    best_string = ' '.join([id2sym(int(x)) for x in best_string.split(' ')])
    print(best_string, np.exp(prob))
