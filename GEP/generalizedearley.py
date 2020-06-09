"""
Created on Jan 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import queue as Queue
from heapq import heappush, heappop

import numpy as np
import nltk.grammar

# class PriorityQueue(object):
#     def __init__(self):
#         self.queue = []
    
#     def put(self, item):
#         heappush(self.queue, item)
    
#     def get(self):
#         return heappop(self.queue)
    
#     def empty(self):
#         return len(self.queue) == 0

#     def __len__(self):
#         return len(self.queue)


class State(object):
    def __init__(self, r, dot, i, j, prefix, prob):
        self._r = r
        self._dot = dot
        self._i = i
        self._j = j
        self._prefix = prefix
        self._prob = prob

    def is_complete(self):
        return self._dot == len(self._r.rhs())

    def next_symbol(self):
        if self.is_complete():
            return None
        return self._r.rhs()[self._dot]

    def __repr__(self):
        rhs = [str(n) for n in self._r.rhs()]
        rhs = ' '.join(rhs[:self._dot]) + " * " + ' '.join(rhs[self._dot:])
        return '[{}:{}:{}] {} -> {} : {:.3f} "{}"'\
            .format(self._dot, self._i, self._j, self._r.lhs(), rhs, self._prob, ' '.join(self._prefix))

    @property
    def r(self): return self._r

    @property
    def dot(self): return self._dot

    @property
    def i(self): return self._i

    @property
    def j(self): return self._j

    @property
    def prefix(self): return self._prefix

    @property
    def prob(self): return self._prob


class GeneralizedEarley(object):
    def __init__(self, grammar):
        self._grammar = grammar
        self._classifier_output = None
        self._total_frame = 0
        self._cached_prob = None
        self._state_set = None
        self._queue = None
        self._prefix_queue = None
        self._max_prob = None
        self._best_l = None
        self._parse_init()

    def _parse_init(self, classifier_output=None):
        self._queue = Queue.PriorityQueue()
        self._prefix_queue = Queue.PriorityQueue()
        # self._queue = PriorityQueue()
        # self._prefix_queue = PriorityQueue()
        self._state_set = [[[]]]
        for r in self._grammar.productions():
            if str(r.lhs()) == 'GAMMA':
                self._state_set[0][0].append(State(r, 0, 0, 0, [], 0.0))
                break
        self._queue.put((1.0 - 1.0, (0, 0, '', self._state_set[0][0])))
        self._max_prob = -np.inf

        if classifier_output is not None:
            if len(classifier_output.shape) != 2:
                raise ValueError('Classifier output shape not recognized, expecting (frame_num, class_num).')
            self._classifier_output = classifier_output
            self._tail_log_prob = np.log(np.max(self._classifier_output, axis=0))[::-1].cumsum()[::-1]

            self._cached_prob = dict()
            self._total_frame = self._classifier_output.shape[0]
            self._class_num = self._classifier_output.shape[1]
            self._cached_prob[''] = np.ones(self._total_frame + 1) * np.finfo('d').min
            # self._cached_prob[''][self._total_frame] = 0.0
            self._cached_prob[''][self._total_frame] = self._tail_log_prob[0]


    def parse(self, classifier_output):
        self._parse_init(classifier_output)
        while not self._queue.empty():
            _, (m, n, set_l, current_set) = self._queue.get()
            branch_probs = dict()
            branch_probs[set_l] = self._cached_prob[set_l][self._total_frame-1]
            for s in current_set:
                l = ' '.join(s.prefix)
                if self._cached_prob[l][self._total_frame-1] > self._max_prob:
                    self._max_prob = self._cached_prob[l][self._total_frame-1]
                    self._best_l = l
                
                if s.is_complete():
                    self.complete(m, n, s)
                elif nltk.grammar.is_nonterminal(s.next_symbol()):
                    self.predict(m, n, s)
                elif nltk.grammar.is_terminal(s.next_symbol()):
                    if m == self._total_frame:
                        continue
                    new_l = self.scan(m, n, s)
                    branch_probs[new_l] = self._cached_prob[new_l][self._total_frame]
                else:
                    raise ValueError('No operation (predict, scan, complete) applies to state {}'.format(s))

            # Early stop
            if not self._queue.empty():
                _, best_prefix_string = self._prefix_queue.get()
                max_prefix_prob = self._cached_prob[best_prefix_string][self._total_frame]
            else:
                max_prefix_prob = - np.inf
            max_branch_prob = max([val for key, val in branch_probs.items()])
            if branch_probs[set_l] == max_branch_prob:
                if max_branch_prob > self._max_prob:
                    self._best_l, self._max_prob = set_l, max_branch_prob
                if self._max_prob > max_prefix_prob:
                    print('Find best parse before exhausting all strings.')  # TODO: check validity
                    return self._best_l, self._max_prob
        return self._best_l, self._max_prob

    def get_log_prob_sum(self):
        log_prob = np.log(self._classifier_output).transpose()
        log_prob_sum = np.zeros((self._class_num, self._total_frame, self._total_frame))
        for c in range(self._class_num):
            for b in range(self._total_frame):
                log_prob_sum[c, b, b] = log_prob[c, b]
        for c in range(self._class_num):
            for b in range(self._total_frame):
                for e in range(b+1, self._total_frame):
                    log_prob_sum[c, b, e] = log_prob_sum[c, b, e-1] + log_prob[c, e]
        return log_prob, log_prob_sum

    def compute_labels(self):
        log_prob, log_prob_sum = self.get_log_prob_sum()

        tokens = [int(token) for token in self._best_l.split(' ')]
        dp_tables = np.zeros((len(tokens), self._total_frame))
        traces = np.zeros_like(dp_tables)

        for end in range(0, self._total_frame):
            dp_tables[0, end] = log_prob_sum[tokens[0], 0, end]

        for token_i, token in enumerate(tokens):
            if token_i == 0:
                continue
            for end in range(token_i, self._total_frame):
                max_log_prob = -np.inf
                for begin in range(token_i, end+1):
                    check_prob = dp_tables[token_i-1, begin-1] + log_prob_sum[token, begin, end]
                    if check_prob > max_log_prob:
                        max_log_prob = check_prob
                        traces[token_i, end] = begin-1
                dp_tables[token_i, end] = max_log_prob

        # Back tracing
        token_pos = [-1 for _ in tokens]
        token_pos[-1] = self._total_frame - 1
        for token_i in reversed(range(len(tokens)-1)):
            token_pos[token_i] = int(traces[token_i+1, token_pos[token_i+1]])

        labels = - np.ones(self._total_frame).astype(np.int)
        labels[:token_pos[0]+1] = tokens[0]
        for token_i in range(1, len(tokens)):
            labels[token_pos[token_i-1]+1:token_pos[token_i]+1] = tokens[token_i]

        return labels, self._best_l.split(' '), token_pos

    def complete(self, m, n, s):
        for back_s in self._state_set[s.i][s.j]:
            if str(back_s.next_symbol()) == str(s.r.lhs()):
                new_s = State(back_s.r, back_s.dot+1, back_s.i, back_s.j, s.prefix, s.prob)
                # # if str(new_s.r.lhs()) == 'GAMMA':
                # #     print(new_s.prefix)

                # # For grammars that don't have recursive rules
                # self._state_set[m][n].append(new_s)

                # For grammars that have recursive rules
                state_exist = False
                for exist_s in self._state_set[m][n]:
                    if str(exist_s) == str(new_s):
                        state_exist = True
                        break
                if not state_exist:
                    # print 'complete: S[{}, {}]'.format(m, n), new_s
                    self._state_set[m][n].append(new_s)

    def predict(self, m, n, s):
        expand_symbol = str(s.next_symbol())
        for r in self._grammar.productions():
            if expand_symbol == str(r.lhs()):
                new_s = State(r, 0, m, n, s.prefix, s.prob)

                # # For grammars that don't have recursive rules
                # self._state_set[m][n].append(new_s)

                # For grammars that have recursive rules
                state_exist = False
                for exist_s in self._state_set[m][n]:
                    if str(exist_s) == str(new_s):
                        state_exist = True
                        break
                if not state_exist:
                    # print 'predict: S[{}, {}]'.format(m, n), new_s
                    self._state_set[m][n].append(new_s)

    def scan(self, m, n, s):
        new_prefix = s.prefix[:]
        new_prefix.append(str(s.next_symbol()))
        prob = self.compute_prob(new_prefix)
        new_s = State(s.r, s.dot+1, s.i, s.j, new_prefix, prob)
        if m == len(self._state_set) - 1:
            new_n = 0
            self._state_set.append([])
        else:
            new_n = len(self._state_set[m+1])

        # To eliminate same prefix branches
        state_exist = False
        for state_set in self._state_set[m+1]:
            exist_s = state_set[0]
            if str(exist_s) == str(new_s):
                state_exist = True
                break

        new_prefix_str = ' '.join(new_prefix)
        if not state_exist:
            # print 'scan: S[{}, {}]'.format(m+1, new_n), new_s
            self._state_set[m+1].append([])
            self._state_set[m+1][new_n].append(new_s)
            self._queue.put((1.0 - prob, (m + 1, new_n, new_prefix_str, self._state_set[m + 1][new_n])))
            self._prefix_queue.put((1.0 - prob, new_prefix_str))

        return new_prefix_str

    def compute_prob(self, prefix):
        l = ' '.join(prefix)
        if l not in self._cached_prob:
            k = int(prefix[-1])
            l_minus = ' '.join(prefix[:-1])
            self._cached_prob[l] = np.ones(self._total_frame + 1) * np.finfo('d').min

            t = len(prefix) - 1
            if t == 0:
                self._cached_prob[l][t] = np.log(self._classifier_output[t, k])
            else:
                self._cached_prob[l][t] = np.log(self._classifier_output[t, k]) + self._cached_prob[l_minus][t-1]
            self._cached_prob[l][self._total_frame] = self._cached_prob[l][t] + self._tail_log_prob[t+1]

            # if len(prefix) == 1:
            #     self._cached_prob[l][0] = np.log(self._classifier_output[0, k])

            # Compute p(l)
            # for t in range(1, self._total_frame):
            #     max_log = max(self._cached_prob[l][t-1], self._cached_prob[l_minus][t-1])
            #     self._cached_prob[l][t] = np.log(self._classifier_output[t, k]) + max_log + np.log(np.exp(self._cached_prob[l][t-1]-max_log) + np.exp(self._cached_prob[l_minus][t-1]-max_log))
            

            # Compute p(l...)
            # if self._total_frame == 1:
            #     max_log = self._cached_prob[l][0]
            # else:
            #     max_log = max(self._cached_prob[l][0], np.max(self._cached_prob[l_minus][0:self._total_frame - 1]))
            # self._cached_prob[l][self._total_frame] = np.exp(self._cached_prob[l][0]-max_log)
            # for t in range(1, self._total_frame):
            #     self._cached_prob[l][self._total_frame] += self._classifier_output[t, k] * np.exp(self._cached_prob[l_minus][t-1]-max_log)
            # self._cached_prob[l][self._total_frame] = np.log(self._cached_prob[l][self._total_frame]) + max_log


        # Search according to prefix probability (Prefix probability stored in the last dimension!!!!!!)
        return self._cached_prob[l][self._total_frame]


def main():
    pass


if __name__ == '__main__':
    main()
