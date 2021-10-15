<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Multi-Stanza: Stanza with Support for Multiple Parses and Error Correction</h2>

# General
## Terminology
A `Document` represents an ordered collection of sentences. This is a document:
````
I went to school today. I learned many things.
````

This is also a document:
````
I went to school today.
````

A `Sentence` represents an ordered sequence of words. A `Word` is a token. These are all tokens:
```
Food
[
hello
.
jealous
'
much
?
```

## Processor
A processor takes a `str` or`Document` object and add its own annotations. Examples or processors are `tok`, `pos`, `lemma`, `depparse`. The `tok` processor takes an `str` and creates a `Document` object with the tokenized text. The `pos` processor takes a `Document` object and adds POS (Part-Of-Speech) annotations to each token. The `lemma` processor takes a `Document` object and adds the lemmatized form of each token. The `depparse` processor takes a document and adds dependency annotations to each token. Most processors depend on the result of other processors. This is where the pipeline, described in the next section, comes in.

Some processors, like `pos`, `lemma`, and `depparse` have the ability to generate multiple results. This is for several reasons. The first is that natural ambiguity is possible for each of the functions that these processors handle. Thus, the processors must return multiple results to have the ability capture the multiple possible meanings. Some examples:
````
Did you see her dress?
British Left waffles on Falkland Islands.
John saw the man with the telescope.
````
For the first sentence, the part-of-speech for `dress` can be either a verb or a noun. For the second sentence, it is possible for `Left` to be a noun, and thus be lemmatized to `Left`. It is also possible for `Left` to be a verb, and thus be lemmatized to `leave`. In the third sentence, it is possible the the man to have the telescope, and thus for `telescope` to be a dependency of `man`. It is also possible that `John` has the telescope, and thus for `telescope` to be a dependency of `John`.

The second reason stems from the fact that processors are neural networks. Hence, despite having high accuracy, it is possible for them to make mistakes. In the sentence `The student protests against the government.`, the best prediction of the `pos` neural tagger labels `protests` as a noun.. This is incorrect. However, the second-best prediction of the `pos` neural tagger labels `protests` as a verb. This is correct. By allowing the processors to generate multiple results, it is more likely that the correct interpretation of the sentence will be generated.

Each processor that has the ability to return multiple `Document` objects also returns a score associated the new annotations added to the `Document` objects. A higher score indicates a higher confidence about the particular set of annotations added to the `Document` object. This allows the pipeline to distinguish which are the annotated documents with the highest confidences, potentially throwing out documents that were labeled with annotations with low confidence.

## Pipeline
Multi-Stana is a pipeline. This means that given an initial string of text, the text is run through several processors. The output from each processor is fed to the the next processor. The processors are ordered in such a way so that processors which depend on results from other processors will be run after those other processors are run. The initial processor is usually the `tok` processor, which takes an `str` object and performs tokenization to generate a `Document` object with the tokenized text. Processors downstream of `tok` all take a `Document` object. If a processor returns multiple `Document` objects, like `pos`, `lemma`, and `depparse` are capable of doing, each of these `Document` objects are fed one by one through the rest of the pipeline. The order the `Document` objects are fed through the rest of the pipeline in descending order by the score of the annotations of the latest processor. 

## Levels
Some processors generate word-level predictions. These are processors like `pos` and `lemma`. In the `pos` processor, a word can have multiple different possible predictions for its Part-Of-Speech with, differing confidences. In the `lemma` processor, a word can have multiple possible predictions for its lemma form, with differing confidences. Some processors generate sentence-level preditions.  These are processors like `depparse`, where the dependency parse for an entire sentence is generated. In the `depparse` processor, a sentence can have multiple predictions for its dependency parse, with differing confidences.

Given a set of POS tags for a *single* word, with differing confidences, it is straightforward to tell which is the best POS tag for the word. This is simply the tag with the highest confidence. It is also straightforward to tell what is the next-best POS tag for the word. This is the is the score with the *next*-highest confidence. However, things are trickier when it comes to the document level. Given a document, where each word is associated with a set of POS tags with differing confidences, what is the best set of POS tags for the document? It seems reasonable to believe that this is simply the highest-confidence POS tag for each word, and this is indeed what `multi-stanza` does by default. What is the *next* best set of POS tags for the document? Here is where levels come into play.

There are three different levels. These levels are `Word`, `Sentence`, and `Document`. The levels
specify what is the largest lexical unit to consider changing the annotations of when attempting to find the next-best set of annotations. 

When using the `Word` level, only the annotations of *a single* word are subject to change when attempting to find the next-best set of annotations. For example, in the sentence `The student protest against the government.`: first the word `The` is considered to have its POS tag changed from its current POS tag to the next-best POS tag. Then, the word `student` is considered to have its POS tag changed from its current POS tag to its next-best POS tag. This continues for every token in the sentence. The changed sentence with the highest aggregate POS tag score is the next one returned.

When using the `Sentence` level, the annotations for *an entire sentence* are subject to change when attempting to find the next-best set of annotations for the document. For example, consider the sentence `I saw the man with the telescope. The telescope was large and grand.` when being processed by the `pos` processor. The first `Document` that would be considered would be the document with the entire first sentence assigned the next-best set of POS tags for that sentence.  The next `Document` to be considered would be the document with the entire second assigned the next-best set of POS tags for that sentence. The changed `Document` with the highest aggregate POS tag score is the next one returned.

When using the `Document` level, the annotations for *an entire document* are subject to change when attempting to find to find the next-best set of annotations for the document. Thus, when finding the next-best set of POS tags for the Document, the next-best POS tag of each word is taken.

Levels are distinct to each processor. It is possible to have the `pos` processor working at the `Word` level, and the `Lemma` processor working at the sentence level. The user specifies that `n` POS predictions should be generated for the `Document`. These are generated by modifying POS tags at the word level. The user also specifies `m` lemma predictions should be generated for the `Document`.  These are generated by taking the (up to) `n` `Document` object that are the output of the `pos` processor, and finding `m` `Document` objects for each of the `n` `Document` objects from the `pos` processor.
