<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Multi-Stanza: Stanza with Support for Multiple Parses and Error Correction</h2>

<div align="center">
    <a href="https://travis-ci.com/stanfordnlp/stanza">
        <img alt="Travis Status" src="https://travis-ci.com/stanfordnlp/stanza.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master">
    </a>
    <a href="https://pypi.org/project/stanza/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/stanza?color=blue">
    </a>
    <a href="https://anaconda.org/stanfordnlp/stanza">
        <img alt="Conda Versions" src="https://img.shields.io/conda/vn/stanfordnlp/stanza?color=blue&label=conda">
    </a>
    <a href="https://pypi.org/project/stanza/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/stanza?colorB=blue">
    </a>
</div>

## Issues and Usage Q&A

To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/giorgianb/multi-stanza/issues). Multi-Stanza is still in active development. Please give feedback on what sort of parsing results you are hoping to get out of multi-stanza so that our multi-stanza can address the challenges of parsing real-world data! Before creating a new issue, please make sure to search for existing issues that may solve your problem, or visit the [Frequently Asked Questions (FAQ) page](https://stanfordnlp.github.io/stanza/faq.html) on our website.

## Installation

### From Source

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of Stanza. For this option, run
```bash
git clone https://github.com/giorgianb/multi-stanza.git
cd stanza
pip install -e .
```

## Running Multi-Stanza

### Getting Started with the neural pipeline

To run your first Stanza pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import stanza
>>> stanza.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English. By default, the default pipeline returns only the best interpretation.
>>> docs = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.") # Returns a list of documents, each containing an interpretation of the sentence
>>> docs[0].sentences[0].print_dependencies() # Print the dependencies of the best interpretation
```

If you encounter `requests.exceptions.ConnectionError`, please try to use a proxy:

```python
>>> import stanza
>>> proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
>>> stanza.download('en', proxies=proxies)  # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en')             # This sets up a default neural pipeline in English
>>> docs = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.") # Returns a list of documents, each containing an interpretation of the sentence
>>> docs[0].sentences[0].print_dependencies() # Print the dependencies of the best interpretation
```

The last command will print out the words in the first sentence in the input string (or [`Document`](https://stanfordnlp.github.io/stanza/data_objects.html#document), as it is represented in Multi-Stanza), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

### Obtaining Multiple Parses
Most languages are not unambiguous parsable. This means there are some sentences within the language for which there are multiple valid interpretations. Consider the sentence:
````
I saw the man with the telescope.
````
For this sentence, it is not clear whether `the man` has the telescope or `I` have the telescope. Both intepretations are plausible. With Multi-Stanza, we can obtain both intepretations:
```python
>>> import stanza
>>> stanza.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en', depparse_n_preds=2) # This sets up a neural pipeline in English. It generates two results at the dependency parsing level.
>>> docs = nlp("I saw the man with the telescope.") # Returns a list of documents, each containing an interpretation of the sentence
>>> docs[0].sentences[0].print_dependencies() # Print the dependencies of the first interpretation. The man has the telescope
('I', 2, 'nsubj')
('saw', 0, 'root')
('the', 4, 'det')
('man', 2, 'obj')
('with', 7, 'case')
('the', 7, 'det')
('telescope', 4, 'nmod')
('.', 2, 'punct')
>>> docs[1].sentences[0].print_dependencies() # Print the dependencies of the second interpretation. I have the telescope!
('I', 2, 'nsubj')
('saw', 0, 'root')
('the', 4, 'det')
('man', 2, 'obj')
('with', 7, 'case')
('the', 7, 'det')
('telescope', 2, 'obl')
('.', 2, 'punct')
```
## LICENSE

Multi-Stanza is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/giorgianb/multi-stanza/blob/multi-pred/LICENSE) file for more details.
