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

## Issues and Usage Q&A

To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/giorgianb/multi-stanza/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem, or visit the [Frequently Asked Questions (FAQ) page](https://stanfordnlp.github.io/stanza/faq.html) on our website.

## Installation

### From Source

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of Stanza. For this option, run
```bash
git clone https://github.com/giorgianb/multi-stanza.git
cd stanza
pip install -e .
```

## Running Stanza

### Getting Started with the neural pipeline

To run your first Stanza pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import stanza
>>> stanza.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

If you encounter `requests.exceptions.ConnectionError`, please try to use a proxy:

```python
>>> import stanza
>>> proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
>>> stanza.download('en', proxies=proxies)  # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en')             # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command will print out the words in the first sentence in the input string (or [`Document`](https://stanfordnlp.github.io/stanza/data_objects.html#document), as it is represented in Stanza), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

See [our getting started guide](https://stanfordnlp.github.io/stanza/installation_usage.html#getting-started) for more details.

## LICENSE

Stanza is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/stanfordnlp/stanza/blob/master/LICENSE) file for more details.
