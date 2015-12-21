# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk


# LANGUAGE = "czech"
LANGUAGE = 'english'
SENTENCES_COUNT = 3

def summarize(text, text_language, num_sentences):
    parser = PlaintextParser(text, Tokenizer(text_language))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(text_language)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(text_language)
    summary = ''
    for sentence in summarizer(parser.document, num_sentences):
        summary += str(sentence)
    return summary

if __name__ == "__main__":
    # nltk.download()
    # url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    url = 'https://en.wikipedia.org/wiki/Hill_climbing'
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    text = "This is a test. I am looking to see that 2 most imp sentences are returned. This call should work. If not, will rework the call."
    parser = PlaintextParser(text, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)