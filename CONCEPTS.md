<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Multi-Stanza: Stanza with Support for Multiple Parses and Error Correction</h2>

# General

## Processor
A processor takes a `str` or`Document` object and add its own annotations. Examples or processors are `tok`, `pos`, `lemma`, `depparse`. The `tok` processor takes an `str` and creates a `Document` object with the tokenized text. The `pos` processor takes a `Document` object and adds Part-of-Speech annotations to each token. The `lemma` processor takes a `Document` object and adds the lemmatized form of each token. The `depparse` processor takes a document and adds dependency annotations to each token. Most processors depend on the result of other processors. This is where the Pipeline comes in.

## Pipeline
Multi-Stana is a pipeline. This means that given an initial string of text, the text is run through several processors.
