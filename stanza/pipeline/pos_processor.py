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

def get_indices(item, recurse=0):
    def helper(item, index):
        if type(item) is list or type(item) is tuple:
            cur_recurse = 0
            if recurse < len(item):
                cur_recurse = recurse
            next_item = item[cur_recurse]
            index.append(len(item))
            return helper(next_item, index)
        else:
            return index

    return helper(item, [])

class NextBest:
    # We want to return the next-best overall prediction
    # For a single sentence out of all of them, we modify the POS
    def __init__(self, preds):
        # preds: (n_pred, n_sentence, n_word, feature, score)
        self._preds = preds
        self._n_preds = len(self._preds)
        self._n_features = len(self._preds[0][0][0])

    def _access(self, fi, index, score=False):
        si = 0 if not score else 1
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

        score /= (total_lengths * nfeatures)

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
        from icecream import ic
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

@register_processor(name=POS)
class POSProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([POS])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path']) if 'pretrain_path' in config else None
        # set up trainer
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        batch = DataLoader(
            document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
            sort_during_eval=True)
        preds = []
        for i, b in enumerate(batch):
            preds.append(self.trainer.predict(b))

        # n_batch, n_pred, n_sentence, n_word, feature

        # Get rid of batch
        n_preds = len(preds[0])
        merged_preds = []
        for i in range(n_preds):
            pred = []
            for pred_minibatch in preds:
                pred.extend(pred_minibatch[i])
            merged_preds.append(pred)
        preds = merged_preds

        docs = []
        serialized = batch.doc.to_serialized()
        for score, pred in itertools.islice(NextBest(preds), n_preds):
            copy = doc.Document.from_serialized(serialized)
            # pred should be (n_sent, n_word, n_feature)
            pred = unsort(pred, batch.data_orig_idx)
            copy.set([doc.UPOS, doc.XPOS, doc.FEATS], [y for x in pred for y in x])
            docs.append(copy)

        return tuple(docs)
