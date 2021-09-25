"""
Processor for performing part-of-speech tagging
"""

from stanza.models.common import doc
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import unsort
from stanza.models.pos.data import DataLoader
from stanza.models.pos.trainer import Trainer
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
import heapq
import itertools
from functools import reduce

from icecream import ic


@register_processor(name=POS)
class POSProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([POS])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path']) if 'pretrain_path' in config else None
        self._n_preds = config.get('n_preds', 3)
        # set up trainer
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu, n_preds=self._n_preds)
        self._next_upos = lambda upos, xpos, ufeats, upi, xpi, ufi: upi + 1
        self._next_xpos = lambda upos, xpos, ufeats, upi, xpi, ufi: xpi + 1
        self._next_ufeats = lambda upos, xpos, ufeats, upi, xpi, ufi: ufi + 1
        self._scorer = lambda ups, xps, ufs: (ups + xps + ufs) / 3

        if config.get('dependency_parse_aware_top_k', 0) == 1:
            # TODO: these have to be verified
            # Look at noun. Is it ok to have it a broad subset (all nouns, including proper)?
            # or a strict one?
            UPOS_MAPPINGS = {
                    'ADJ': {'JJ', 'JJR', 'JJS'},
                    'ADP': {'IN', 'RP'},
                    'ADV': {'RB', 'RBR', 'RBS'},
                    'AUX': {'MD'},
                    'CCONJ': {'CC'},
                    'DET': {'DT', 'PDT'},
                    'INTJ': {'UH'},
                    'NOUN': {'NN', 'NNS', 'NNP', 'NNPS'},
                    'NUM': {'CD'},
                    'PART': {'TO', 'POS'},
                    'PRON': {'PRP', 'PRP$', 'WP', 'WDT', 'EX', 'WP$'},
                    'PROPN': {'NNP', 'NNPS'},
                    'PUNCT': {'.'},
                    'SCONJ': {'IN', 'WRB'},
                    'SYM': {'SYM'},
                    'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
                    'X': {'FW', 'LS'}
            }
            def next_xpos(upos, xpos, ufeats, upi, xpi, ufi):
                # get what would be the next upos tag
                next_upos = self._next_upos(upos, xpos, ufeats, upi, xpi, ufi)
                # There's no more upos tags so we can't match
                if next_upos >= len(upos):
                    return len(xpos) # This indicates no more options

                # this is the set of XPOS tags we'll accept
                options_set = UPOS_MAPPINGS[upos[next_upos][0]]
                # this is the set of XPOS tags we can actually use
                xpos_options = tuple(filter(lambda x: x[1][0] in options_set, enumerate(xpos)))
                # are there actually any tags?
                if len(xpos_options) == 0:
                    return len(xpos)

                # return the highest-scoring XPOS tag that matches the UPOS tag
                return xpos_options[0][0]
            self._next_xpos = next_xpos
            # Ignore XPOS score
            # TODO: figure out how to enable XPOS score
            self._scorer = lambda ups, xps, ufs: (ups + ufs) / 2

        elif config.get('dependency_parse_aware_top_k', 0) == 2:
            EQUIVALENCE_CLASSES = (
                    {'JJ', 'JJR', 'JJS'},
                    {'NN', 'NNS', 'NNP', 'NNPS'},
                    {'PRP', 'PRP$'},
                    {'RB', 'RBR', 'RBS'},
                    {'VB', 'VBD', 'VBG', 'VBN', 'VBP'},
                    {'WP', 'WP$', 'WRB'}
            )
            NEED_CHECKED = reduce(lambda a, b: a | b, EQUIVALENCE_CLASSES)
            MAPPING = {t:c for c in EQUIVALENCE_CLASSES for t in c}
            def next_xpos(upos, xpos, ufeats, upi, xpi, ufi):
                if xpos[xpi][0] not in NEED_CHECKED:
                    return xpi + 1

                c = MAPPING[xpos[xpi][0]]
                for i in range(xpi + 1, len(xpos)):
                    if xpos[i][0] not in c:
                        return i

                return len(xpos) # This indicates that there are no more options
            self._next_xpos = next_xpos

    def process(self, document):
        batch = DataLoader(
            document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
            sort_during_eval=True)
        preds = []
        for i, b in enumerate(batch):
            preds.append(self.trainer.predict(b))

        # n_batch, n_pred, n_sentence, n_word, feature

        # Get rid of batch
        merged_preds = []
        n_preds = min(self._n_preds, len(preds[0]))
        for i in range(n_preds):
            pred = []
            for pred_minibatch in preds:
                pred.extend(pred_minibatch[i])
            merged_preds.append(pred)
        preds = merged_preds

        docs = []
        scores = []
        serialized = batch.doc.to_serialized()
        nb = NextBest(preds, self._scorer, self._next_upos, self._next_xpos, self._next_ufeats)
        for score, pred in itertools.islice(nb, self._n_preds):
            copy = doc.Document.from_serialized(serialized)
            # pred should be (n_sent, n_word, n_feature)
            pred = unsort(pred, batch.data_orig_idx)
            copy.set([doc.UPOS, doc.XPOS, doc.FEATS], [y for x in pred for y in x])
            copy.set([doc.POS_SCORE], [score], to_document=True)
            docs.append(copy)
            scores.append(score)

        return tuple(docs) #, scores TODO: return scores eventually

class NextBest:
    def __init__(self, preds, scorer, next_upos, next_xpos, next_ufeats):
        # preds: (n_pred, n_sentence, n_word, feature, score)
        self._preds = preds
        self._n_preds = len(self._preds)
        self._n_sents = len(self._preds[0])
        self._upos = self._extract_feature(preds, 0)
        self._xpos = self._extract_feature(preds, 1)
        self._ufeats = self._extract_feature(preds, 2)
        self._scorer = scorer
        self._next_upos = next_upos
        self._next_xpos = next_xpos
        self._next_ufeats = next_ufeats

    @staticmethod
    def _extract_feature(pred, fi):
        return [[[pred[i][j][k][fi] for k in range(len(pred[i][j]))] for j in range(len(pred[i]))] for i in range(len(pred))]

    @staticmethod
    def _pack_features(upos, xpos, ufeats):
        # Expects extracted upos, xpos, ufeats from _accesss
        # Should NOT have score index
        return [[(upos[i][j], xpos[i][j], ufeats[i][j]) for j in range(len(upos[i]))] for i in range(len(upos))]

    def _access(self, feat, index, score=False):
        si = 1 if score else 0
        return [[feat[index[i][j]][i][j][si] for j in range(len(index[i]))] for i in range(len(index))]

    def _score(self, upos_index, xpos_index, ufeats_index):
        score = 0
        total_lengths = 0
        upos = self._access(self._upos, upos_index, score=True)
        xpos = self._access(self._xpos, xpos_index, score=True)
        ufeats = self._access(self._ufeats, ufeats_index, score=True)
        feats = self._pack_features(upos, xpos, ufeats)
        for sent in feats:
            score += sum(map(lambda x: self._scorer(*x), sent))
            total_lengths += len(sent)

        score /= total_lengths

        return score

    def __iter__(self):
        # What does an index look like?
        # index is (n_sentence, n_word)
        # Use simple summation to combine score across features
        self._seen = set()
        start_index = tuple(tuple(0 for i in range(len(sent))) for sent in self._upos[0])
        score = self._score(start_index, start_index, start_index)
        self._tie_counter = 0
        tc = self._tie_counter
        self._tie_counter += 1
        self._queue = [(-score, tc, start_index, start_index, start_index)]
        self._seen.add((start_index, start_index, start_index))

        return self

    def _next_wrapper(self, next_feat, upos_i, xpos_i, ufeats_i, i, j):
        upos = [self._upos[k][i][j] for k in range(len(self._upos))]
        xpos = [self._xpos[k][i][j] for k in range(len(self._upos))]
        ufeats = [self._ufeats[k][i][j] for k in range(len(self._upos))]
        return next_feat(upos, xpos, ufeats, upos_i[i][j], xpos_i[i][j], ufeats_i[i][j])
        

    def __next__(self):
        if len(self._queue) == 0:
            raise StopIteration

        score_ret, _, upos_index_ret, xpos_index_ret, ufeats_index_ret = heapq.heappop(self._queue)
        for i in range(self._n_sents):
            n_words = len(self._preds[0][i])
            for j in range(n_words):
                next_upos = self._next_wrapper(self._next_upos, upos_index_ret, xpos_index_ret, ufeats_index_ret, i, j)
                if next_upos >= self._n_preds:
                    continue

                next_xpos = self._next_wrapper(self._next_xpos, upos_index_ret, xpos_index_ret, ufeats_index_ret, i, j)
                if next_xpos >= self._n_preds:
                    continue

                next_ufeats = self._next_wrapper(self._next_ufeats, upos_index_ret, xpos_index_ret, ufeats_index_ret, i, j)
                if next_ufeats >= self._n_preds:
                    continue

                new_upos_index = [list(sent) for sent in upos_index_ret]
                new_upos_index[i][j] = next_upos
                new_upos_index = tuple(tuple(sent) for sent in new_upos_index)
                new_xpos_index = [list(sent) for sent in xpos_index_ret]
                new_xpos_index[i][j] = next_xpos
                new_xpos_index = tuple(tuple(sent) for sent in new_xpos_index)
                new_ufeats_index = [list(sent) for sent in ufeats_index_ret]
                new_ufeats_index[i][j] = next_ufeats
                new_ufeats_index = tuple(tuple(sent) for sent in new_ufeats_index)

                new_index = (new_upos_index, new_xpos_index, new_ufeats_index)
                if new_index in self._seen:
                    continue

                self._seen.add(new_index)
                new_score = self._score(new_upos_index, new_xpos_index, new_ufeats_index)
                tc = self._tie_counter
                self._tie_counter += 1
                heapq.heappush(self._queue, (-new_score, tc, new_upos_index, new_xpos_index, new_ufeats_index))

        upos_ret = self._access(self._upos, upos_index_ret)
        xpos_ret = self._access(self._xpos, xpos_index_ret)
        ufeats_ret = self._access(self._ufeats, ufeats_index_ret)

        return -score_ret, self._pack_features(upos_ret, xpos_ret, ufeats_ret)

class SimpleNextBest:
    # We want to return the next-best overall prediction
    # For a single sentence out of all of them, we modify the POS
    def __init__(self, preds):
        # preds: (n_pred, n_sentence, n_word, feature, score)
        self._preds = preds
        self._n_preds = len(self._preds)
        self._n_features = len(self._preds[0][0][0])

    def _access(self, fi, index, score=False):
        si = 1 if score else 0
        return [[self._preds[index[i][j]][i][j][fi][si] for j in range(len(index[i]))] for i in range(len(index))]

    def _ret_access(self, index):
        return [[[self._preds[index[i][j]][i][j][fi][0] for fi in range(self._n_features)] for j in range(len(index[i]))] for i in range(len(index))]


    def _score(self, index):
        # Simple summation across all scores and all features
        score = 0
        acc = lambda fi: self._access(fi, index, score=True)
        total_lengths = 0
        nfeatures = self._n_features
        for fi in range(nfeatures):
            for sent in acc(fi):
                score += sum(sent)
                total_lengths += len(sent)

        score /= total_lengths

        return score


    def __iter__(self):
        # What does an index look like?
        # index is (n_sentence, n_word)
        # Use simple summation to combine score across features
        self._seen = set()
        start_index = tuple(tuple(0 for i in range(len(sent))) for sent in self._preds[0])
        score = self._score(start_index)
        self._tie_counter = 0
        tc = self._tie_counter
        self._tie_counter += 1
        self._queue = [(-score, tc, start_index)]
        self._seen.add(start_index)

        return self

    def __next__(self):
        if len(self._queue) == 0:
            raise StopIteration

        score_ret, _, index_ret = heapq.heappop(self._queue)
        n_sentences = len(self._preds[0])
        for i in range(n_sentences):
            n_words = len(self._preds[0][i])
            assert len(index_ret[i]) == len(self._preds[0][i])
            for j in range(n_words):
                if index_ret[i][j] + 1 >= self._n_preds:
                    continue

                new_index = [list(sent) for sent in index_ret]
                new_index[i][j] += 1
                new_index = tuple(tuple(sent) for sent in new_index)

                if new_index in self._seen:
                    continue

                self._seen.add(new_index)
                new_score = self._score(new_index)
                tc = self._tie_counter
                self._tie_counter += 1
                heapq.heappush(self._queue, (-new_score, tc, new_index))

        return -score_ret, self._ret_access(index_ret)
