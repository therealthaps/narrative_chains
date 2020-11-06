#!/usr/bin/env python3

import pandas as pd
import neuralcoref
import spacy
import random
from collections import namedtuple, Counter
import tqdm

nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)

ParsedStory = namedtuple("ParsedStory", "id title story one two three four five".split())

def load_data(rocstories):
    """Loads in rocstories csv file and iterates over the stories
    - Returns a generator of stories
    - a generator is like a collection/list that isn't populated until a item is requested
    - calling a ~next~ function on the generator gives an item if there is one left
    - if there are no items left, returns that it is empty
    """
    data = pd.read_csv(rocstories)
    return data.itertuples()

def parse_story(story):
    """Take a single story and run it through spacy"""
    name = story.storytitle
    # this is very compressed
    sentences = [nlp(sentence) for sentence in story[3:]]
    full_story = nlp(" ".join(story[3:]))
    return ParsedStory(story.storyid, name, full_story, *sentences)

def parse_story_verbose(story):
    """Same as parse_story but this is longer"""
    name = story.storytitle
    # sentences = [nlp(sentence) for sentence in story[3:]]
    just_story = story[3:]
    sentences = []
    for sentence in just_story:
        sentences.append(nlp(sentence))
    # sentences is a list
    full_story = nlp(" ".join(story[3:]))
    return ParsedStory(name, full_story, sentences[0], sentences[1], sentences[2], sentences[3], sentences[4])


def take_sample(gen, sample=None, replacement=0.2):
    if not sample:
        """If we don't have a sample specified, return the entire generator"""
        return gen
    # take the first ~sample~ number of stories
    # randomly replace stories as we iterate
    sentences = []

    for x in gen:
        sentences.append(x)
        if len(sentences) == sample:
            break
    # sentences has ~sample~ sentences in it
    for x in gen:
        if random.random() <= replacement:
            index = random.randint(1, sample) - 1
            sentences[index] = x
    return tqdm.tqdm(sentences)

def process_corpus(rocstories, sample=None, replacement=0.2):
    """rocstories is a string with the path to the rocstories.csv file
    sample: load a random sample of ~sample~ sentences
    """
    # 1 load the data
    # 2 iterate over everything
    #   3: parse everything
    data = load_data(rocstories)
    dataset = [parse_story(story) for story in take_sample(data, sample, replacement)]

    for story in dataset:
        process_story(story)

    return dataset

def process_story(parsedstory, heuristic=2, verbose=True):
    prot = protagonist(parsedstory, heuristic=2)
    dep_pairs = extract_dependency_pairs(parsedstory)
    # TODO: Next step: count dependency pairs

    if verbose:
        print("------------")
        print("Protagonist is: ", prot)
        print("story: ", parsedstory.title)
        for sentence in parsedstory[-5:]:
            print(sentence)
        for d in dep_pairs:
            print("\t", d[1:])
        print("------------")

def dereference_pair(token, story):
    """returns a set id for an entity per story"""
    if token.text.lower() in ["i", "me"]:
        return -1
    pool = story._.coref_clusters
    for ref in pool:
        for mention in ref.mentions:
            if token == mention.root:
                return ref.i
    return None

def extract_dependency_pairs(parse):
    """Get a story, return a list of dependency pairs"""
    verbs = [verb for verb in parse.story if verb.pos_ == "VERB"]
    deps = []
    for verb in verbs:
        for child in verb.children:
            entity_index = dereference_pair(child, parse.story)
            tup = (parse.id, verb, child.dep_, child, entity_index)
            if entity_index != None:
                deps.append(tup)
    return deps


# Protagonist detection
def protagonist(story, heuristic=2):
    story = story.story
    if heuristic == 1:
        return protagonist_heuristic_one(story)
    elif heuristic == 2:
        return protagonist_heuristic_two(story)
    elif heuristic == 3:
        raise NotImplementedError


# Heuristic 1: first entity
def protagonist_heuristic_one(story):
    """Story is parsed by spacy"""
    return [(story.ents[0].text, 1)]

# Heuristic 2: most frequently mentioned entity
def protagonist_heuristic_two(story):
    #1 get entities
    if not story._.coref_clusters:
        return None
    return max(story._.coref_clusters, key=lambda cluster: len(cluster.mentions))
