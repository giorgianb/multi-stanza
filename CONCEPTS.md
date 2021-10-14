<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Multi-Stanza: Stanza with Support for Multiple Parses and Error Correction</h2>

# General

## Processor
A processor takes a `str` or`Document` object and add its own annotations. Examples or processors are `tok`, `pos`, `lemma`, `depparse`. The `tok` processor takes an `str` and creates a `Document` object with the tokenized text. The `pos` processor takes a `Document` object and adds Part-of-Speech annotations to each token. The `lemma` processor takes a `Document` object and adds the lemmatized form of each token. The `depparse` processor takes a document and adds dependency annotations to each token. Most processors depend on the result of other processors. This is where the Pipeline comes in.

Some processors, like `pos`, `lemma`, and `depparse` have the ability to generate multiple results. This is for several reasons. The first is that natural ambiguity is possible for each of the functions that these processors handle. Thus, the processors must return multiple results to have the ability capture the multiple possible meanings. Some examples:
````
Did you see her dress?
British Left waffles on Falkland Islands.
John saw the man with the telescope.
````
For the first sentence, the part-of-speech for `dress` can be either a verb or a noun. For the second sentence, it is possible for `Left` to be a noun, and thus be lemmatized to `Left`. It is also possible for `Left` to be a verb, and thus be lemmatized to `leave`. In the third sentence, it is possible the the man to have the telescope, and thus for `telescope` to be a dependency of `man`. It is also possible that `John` has the telescope, and thus for `telescope` to be a dependency of `John`.

The second reason stems from the fact that processors are neural networks. Hence, despite having high accuracy, it is possible for them to make mistakes. In the sentence `The student protests against the government.`, the best prediction of the `pos` neural tagger labels `protests` as a noun.. This is incorrect. However, the second-best prediction of the `pos` neural tagger labels `protests` as a verb. This is correct. By allowing the processors to generate multiple results, it is more likely that the correct interpretation of the sentence will be generated.

Each processor that has the ability to return multiple `Document` objects also returns a score associated with each of the `Document` objects. A higher score indicates a higher confidence about the particular set of annotations added to the `Document` object. This allows the pipeline to distinguish which are the annotated documents with the highest confidences, potentially throwing out documents that were labeled with annotations with low confidence.

## Pipeline
Multi-Stana is a pipeline. This means that given an initial string of text, the text is run through several processors. The output from each processor is fed to the the next processor. The initial processor is usually the `tok` processor, which takes an `str` object and performs tokenization to generate a `Document` object with the tokenized text. Processors downstream of `tok` all take a `Document` object. If a processor returns multiple `Document` objects, like `pos`, `lemma`, and `depparse` are capable of doing, each of these `Document` objects are fed one by one through the rest of the pipeline. The order the `Document` objects are fed through the rest of the pipeline in descending order by the score of the annotations of the latest processor. 


