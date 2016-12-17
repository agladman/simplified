#!/usr/bin/env Python3

"""
A simplified version of the code to test whether using POSifiedText()
before exporting to JSON affects the output from a reconstituted text model.

Comment out either line 35 or 36 to try with or without POSifiedText().
"""

import json
import markovify
import nltk
import re


class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = ["::".join(tag) for tag in nltk.pos_tag(words)]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


def export_text_model():
    """
    Would usually pull input_text as lines from postgres db but have captured a
    sample to use here in a text file.
    """
    with open('input_text.txt', 'r') as f:
        input_text = f.read()

    text_model = POSifiedText(input_text)
    # text_model = markovify.Text(input_text)
    model_json = text_model.to_json()

    with open('text_model.json', 'w') as json_file:
        json_file.write(json.dumps(model_json))


def create_sentence():
    with open('text_model.json', 'r') as json_file:
        model_data = json.loads(json_file.read())
    reconstituted_model = markovify.Text.from_json(model_data)
    print(reconstituted_model.make_short_sentence(140))


def main():
    export_text_model()
    create_sentence()


if __name__ == '__main__':
    main()
