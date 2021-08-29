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

        # Rearrange to (n_pred, n_document)
        n_preds = len(preds[0])
        rearranged_preds = []
        for i in range(n_preds):
            pred = []
            for pred_group in preds:
                pred.extend(pred_group[i])
            rearranged_preds.append(pred)
        preds = rearranged_preds

        docs = []
        serialized = batch.doc.to_serialized()
        for pred in preds:
            copy = doc.Document.from_serialized(serialized)
            pred = unsort(pred, batch.data_orig_idx)
            copy.set([doc.UPOS, doc.XPOS, doc.FEATS], [y for x in pred for y in x])
            docs.append(copy)

        return tuple(docs)
