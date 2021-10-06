import stanza
from icecream import ic

def next_upos(upos, xpos, ufeats, upi, xpi, ufi, wi):
    # upos is the ENTIRE list of predicted upos tags for the word
    # A list of tuples, each tuple of the form (upos_tag, tag_score)
    # xpos is the ENTIRE list of predicted xpos tags for the word
    # A list of tuples, each tuple of the form (xpos_tag, tag_score)
    # ufeats is the ENTIRE list of predicted ufeats tags for the word
    # A list of tuples, each tuple of the form (ufeats_tag, tag_score)
    # upi is the index of the CURRENT upos tag for the word
    # xpi is the index of the CURRENT xpos tag for the word
    # ufi is the index of the CURRENT ufeats tag for the word
    if wi == (0, 9):
        ic(tuple(filter(lambda x: x[0] in {'VERB', 'NOUN'}, upos)))

    # These are all upos options with a lower score than our current one
    next_upos_options = upos[upi + 1:]

    # Check if the VERB tag is an option
    valid_upos_options = tuple(filter(lambda tag_score: tag_score[0] == 'VERB', next_upos_options))

    if len(valid_upos_options) == 0:
        return len(upos) + 1 # sentinal value indicating we're out of UPOS tags

    return upos.index(valid_upos_options[0]) # Return the index of the next-best UPOS options


# Specifying pos_next_upos overrides the default next_upos of pos_top_k_mode=1
# The default next_upos was simply to return upi + 1
nlp = stanza.Pipeline('en', pos_n_preds=2, pos_top_k_mode=1, pos_next_upos=next_upos, lemma_n_preds=1, depparse_n_preds=1)

docs = nlp('When did the films written by [MASK] writers release?')
print(docs[1]) # This is the desired interpretation
# Note that EVERY set of POS tags past the first one will now be guaranteed to have at least ONE 
# change of a non-verb POS tag to a verb POS tag.
