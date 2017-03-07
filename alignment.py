# Brendan King
# CSE 427
# Homework 1

import numpy as np
import random
import requests
import json, sys


# Constants
UP = 'UP'
LEFT = 'LEFT'
DIAG = 'DIAG'
UNIPROT_BASE_URL = 'http://www.uniprot.org/uniprot/'


def get_score(aa1, aa2):
    return 1 if aa1.capitalize() == aa2.capitalize() else -1

class SmithWaterman:
    def __init__(self, seq1, seq2):
        self.s1 = seq1
        self.s2 = seq2
        self.scores = None

    def score_matrix(self):
        """
        :return: a scoring matrix for this sequence
        """
        scores = np.zeros((len(self.s1) + 1, len(self.s2) + 1))
        for i in range(1, scores.shape[0]):
            for j in range(1, scores.shape[1]):
                scores[i][j] = self._calculate_entry(scores, i, j)
        self.scores = scores
        return scores

    def _calculate_entry(self, scores, i, j):
        match_value = get_score(self.s1[i - 1], self.s2[j - 1]) + scores[i - 1][j - 1]
        left_gap = get_score(self.s1[i - 1], '*') + scores[i - 1][j]
        right_gap = get_score('*', self.s2[j - 1]) + scores[i][j - 1]
        return max(match_value, left_gap, right_gap, 0)

    def get_optimal_alignments(self, scores=None):
        """
        return optimal alignments [((align_seq1, align_seq2), (x, y)), ...]
        """
        if scores is None:
            scores = self.scores if self.scores is not None else self.score_matrix()
        x, y = np.where(scores == np.amax(scores))
        alignments = []
        max_scores = zip(x, y)
        for p in max_scores:
            alignments.extend(self._traceback(scores, p))
        return alignments

    def _traceback(self, scores, p):
        results = list()
        queue = [(p, ('', ''))]
        while len(queue) > 0:
            (x, y), alignment = queue.pop()
            if scores[x][y] == 0.0:
                results.append((alignment, (x, y)))
                continue
            neighbors = [((x - 1, y - 1), DIAG, get_score(self.s1[x - 1], self.s2[y - 1])),
                         ((x - 1, y), UP, get_score(self.s1[x - 1], '*')),
                         ((x, y - 1), LEFT, get_score('*', self.s2[y - 1]))]
            legal_neighbors = [n for n in neighbors if scores[n[0]] + n[2] == scores[(x, y)]]
            max_score = max([scores[n[0]] for n in legal_neighbors])
            max_neighbors = [n for n in legal_neighbors if scores[n[0]] == max_score]
            for n in max_neighbors:
                if n[1] == UP:
                    queue.insert(0, (n[0], (self.s1[x - 1] + alignment[0], '-' + alignment[1])))
                elif n[1] == LEFT:
                    queue.insert(0, (n[0], ('-' + alignment[0], self.s2[y - 1] + alignment[1])))
                else:
                    queue.insert(0, (n[0], (self.s1[x - 1] + alignment[0], self.s2[y - 1] + alignment[1])))
        return results

    def empirical_p_value(self, score, n=10000):
        """
        Generates an empirical p-value based on sampling a distribution of scores from random permutations of one of the
        supplied sequences
        :param score: socre to get p-value of
        :param seq1: first sequence
        :param seq2: second sequence
        :param n: (optional) number of permutations to use
        :return: p-value
        """
        rand_seq = list(self.s2)
        k = 0
        for i in range(0, n):
            random.shuffle(rand_seq)
            rand_str = ''.join(rand_seq)
            sm = SmithWaterman(self.s1, rand_str)
            rand_score = np.amax(sm.score_matrix())
            if rand_score > score:
                k += 1
        return '{:.2e}'.format((float(k) + 1.) / (n + 1.))


class NeedlemanWunsch:
    """
    class for running the Needleman-Wunsch algorithm on two sequences
    """
    def __init__(self, seq1, seq2):
        self.s1 = seq1
        self.s2 = seq2
        self.scores = None

    def score_matrix(self):
        """
        :return: a scoring matrix for this sequence
        """
        scores = np.zeros((len(self.s1) + 1, len(self.s2) + 1))
        # Initialization:
        for i in range(1, scores.shape[0]):
            scores[i][0] = i * get_score('*', self.s1[i - 1])
        for i in range(1, scores.shape[1]):
            scores[0][i] = i * get_score('*', self.s2[i - 1])
        for i in range(1, scores.shape[0]):
            for j in range(1, scores.shape[1]):
                scores[i][j] = self._calculate_entry(scores, i, j)
        self.scores = scores
        return scores

    def _calculate_entry(self, scores, i, j):
        match_value = get_score(self.s1[i - 1], self.s2[j - 1]) + scores[i - 1][j - 1]
        left_gap = get_score(self.s1[i - 1], '*') + scores[i - 1][j]
        right_gap = get_score('*', self.s2[j - 1]) + scores[i][j - 1]
        return max(match_value, left_gap, right_gap)

    def get_optimal_alignments(self, scores=None):
        """
        return optimal alignments [((align_seq1, align_seq2), (x, y)), ...]
        """
        if scores is None:
            scores = self.scores if self.scores is not None else self.score_matrix()
        x, y = len(self.s1), len(self.s2)
        alignments = []
        alignments.extend(self._traceback(scores, (x, y)))
        return alignments

    def _traceback(self, scores, p):
        """
        Use BFS to traceback all optimal alignments through the score matrix
        :param scores:
        :param p:
        :return: aligments [((align_seq1, align_seq2), (x, y)), ...]
        """
        results = list()
        queue = [(p, ('', ''))]
        while len(queue) > 0:
            (x, y), alignment = queue.pop()
            if x == 0 and y == 0:
                results.append((alignment, (x, y)))
                continue
            neighbors = [((x - 1, y - 1), DIAG, get_score(self.s1[x - 1], self.s2[y - 1])),
                         ((x - 1, y), UP, get_score(self.s1[x - 1], '*')),
                         ((x, y - 1), LEFT, get_score('*', self.s2[y - 1]))]
            legal_neighbors = [n for n in neighbors if scores[n[0]] + n[2] == scores[(x, y)]]
            max_score = max([scores[n[0]] for n in legal_neighbors])
            max_neighbors = [n for n in legal_neighbors if scores[n[0]] == max_score]
            for n in max_neighbors:
                if n[1] == UP:
                    queue.insert(0, (n[0], (self.s1[x - 1] + alignment[0], '-' + alignment[1])))
                elif n[1] == LEFT:
                    queue.insert(0, (n[0], ('-' + alignment[0], self.s2[y - 1] + alignment[1])))
                else:
                    queue.insert(0, (n[0], (self.s1[x - 1] + alignment[0], self.s2[y - 1] + alignment[1])))
        return results

    def empirical_p_value(self, score, n=10000):
        """
        Generates an empirical p-value based on sampling a distribution of scores from random permutations of one of the
        supplied sequences
        :param score: socre to get p-value of
        :param seq1: first sequence
        :param seq2: second sequence
        :param n: (optional) number of permutations to use
        :return: p-value
        """
        rand_seq = list(self.s2)
        k = 0
        for i in range(0, n):
            random.shuffle(rand_seq)
            rand_str = ''.join(rand_seq)
            nw = NeedlemanWunsch(self.s1, rand_str)
            rand_score = np.amax(nw.score_matrix())
            if rand_score > score:
                k += 1
        return '{:.2e}'.format((float(k) + 1.) / (n + 1.))


def pretty_print_alignment(alignment, name1='Sequence1', name2='Sequence2'):
    """

    :param alignment: an alignment is a tuple, where the first element is a tuple of aligned sequences (s1, s2)
                      and the second element is a tuple indicating begining indices (x, y) e.g. ((s1, s2), (x, y))
    :param name1: name of first sequence
    :param name2: name of second sequence
    :return:
    """
    ((seq1, seq2), (x, y)) = alignment  # Pattern Match
    k = 0
    while k < len(seq1):
        x_str = str(x + k)
        y_str = str(y + k)
        print str(name1) + ':', x_str, '   ', seq1[k: min(k + 60, k + len(seq1))], '\n'
        print str(name2) + ':', y_str, '   ', seq2[k: min(k + 60, k + len(seq1))], '\n'
        k += 60


def pairwise_scores(list_of_sequences):
    """
    Given a list of sequences 0..n, return an upper triangular matrix where entry (i, j) is the best alignment score
    (SmithWaterman) for the ith and jth sequence in the list
    :return: an n by n upper triangular matrix of alignment scores
    """
    size = len(list_of_sequences)
    scores = np.zeros((size, size))
    for i in range(0, size):
        for j in range(i, size):
            sm = SmithWaterman(list_of_sequences[i], list_of_sequences[j])
            scores[i][j] = np.amax(sm.score_matrix())
    return scores

def pairwise_scores_nw(list_of_sequences):
    """
    Given a list of sequences 0..n, return an upper triangular matrix where entry (i, j) is the best alignment score
    (SmithWaterman) for the ith and jth sequence in the list
    :return: an n by n upper triangular matrix of alignment scores
    """
    size = len(list_of_sequences)
    scores = np.zeros((size, size))
    for i in range(0, size):
        for j in range(i, size):
            sm = NeedlemanWunsch(list_of_sequences[i], list_of_sequences[j])
            scores[i][j] = np.amax(sm.score_matrix())
    return scores

def download_sequence_by_id(uniprot_id):
    """
    Given an accession ID for unitport, download and parse the protein sequence as a string

    Example:

    In [27]: alignment.download_sequence_by_id('P15172')

    Out[27]: u'MELLSPPLRDVDLTAPDGSLCSFATTDDFYDDPCFDSPDLRFFEDLDPRLMHVGALLKPEEHSHFPAAVHPAPGAREDEHVRAPSGHHQAGR
               CLLWACKACKRKTTNADRRKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQALLRDQDAAPPGAAAAFYAPGPL
               PPGRGGEHYSGDSDASSPRSNCSDGMMDYSGPPSGARRRNCYEGAYYNEAPSEPRPGKSAAVSSLDCLSSIVERISTESPAAPALLLADVPS
               ESPPRRQEAAAPSEGESSGDPTQSPDAAPQCPAGANPNPIYQVL'

    *output is a single line

    :param uniprot_id: Accession ID for protein sequence (e.g. P15172)
    :return: sequence as a string
    """
    return _parse_fasta(_download_fasta(uniprot_id))


def _download_fasta(uniprot_id):
    return requests.get(UNIPROT_BASE_URL + str(uniprot_id) + '.fasta').text


def _parse_fasta(fasta_str):
    return ''.join(fasta_str.split('\n')[1:])

if __name__ == "__main__":
    seqs = json.loads(sys.argv[1])
    print json.dumps(pairwise_scores(seqs).tolist())
