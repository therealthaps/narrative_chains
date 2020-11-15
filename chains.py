# !/usr/bin/env python3

import pandas as pd
import neuralcoref
import spacy
import random, math
from collections import namedtuple, Counter, defaultdict as ddict
import tqdm
from pprint import pprint

# Choose your model here
# Recall that "en_core_web_sm" performed significantly worse with regards to coreference resolution
nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)



# This is a magic number
# When we are computing probabilitieis, we don't want anything to have occurred 0 times because some of our functions are undefined around 9
# To avoid 0's, we can use something called Plus One Smoothing, which simply assumes that there is at least a minimum amount of everything
# Obviously, this is not the most accurate thing, since there are almost certainly sentences that never have or will be said
# But it is very useful for estimating probabilities

PLUS_ONE_SMOOTHING = 0.01

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

# I've commented out this function to avoid confusion
# This was just a demonstration
#def parse_story_verbose(story):
#    """Same as parse_story but this is longer"""
#    name = story.storytitle
#    # sentences = [nlp(sentence) for sentence in story[3:]]
#    just_story = story[3:]
#    sentences = []
#    for sentence in just_story:
#        sentences.append(nlp(sentence))
#    # sentences is a list
#    full_story = nlp(" ".join(story[3:]))
#    return ParsedStory(name, full_story, sentences[0], sentences[1], sentences[2], sentences[3], sentences[4])


def take_sample(gen, sample=None, replacement=0.2):
    if not sample:
        """If we don't have a sample specified, return the entire generator"""
        return tqdm.tqdm(gen)
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
    story_counter = ddict(list)

    for story in dataset:
        process_story(story, story_counter=story_counter)

    return dataset, ProbabilityTable(story_counter)

def process_story(parsedstory, heuristic=2, verbose=True, story_counter=ddict(list)):
    prot = protagonist(parsedstory, heuristic=2)
    parse_id, dep_pairs = extract_dependency_pairs(parsedstory)

    story_counter[parse_id] = dep_pairs

    if verbose:
        print("------------")
        print("Protagonist is: ", prot)
        print("story: ", parsedstory.title)
        for sentence in parsedstory[-5:]:
            print(sentence)
        for entity in dep_pairs:
            for d in dep_pairs[entity]:
                print("\t", d)
        print("------------")

# Return a per-story entity-id for a token
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

#Return a list of dependency pairs
def extract_dependency_pairs(parse):
    """Get a story, return a list of dependency pairs"""
    verbs = [verb for verb in parse.story if verb.pos_ == "VERB"]
    deps = ddict(list)
    for verb in verbs:
        for child in verb.children:
            entity_index = dereference_pair(child, parse.story)
            # Add the word/dependency pair to the identified entity
            tup = (verb.lemma_, child.dep_)
            if entity_index != None:
                deps[entity_index].append(tup)
    return parse.id, deps

def coreferring_pairs(parse, token):
    """Because I'm nice, here's a function that gets all the dependency pairs that corefer to the given token
    This is inefficiently based on extract_dependency_pairs and dereference_pair for a reason.
    """
    extracted = extract_dependency_pairs(parse)
    res = dereference_pair(token, parse.story)
    if res is None:
        return []
    return extracted[res]

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

class ProbabilityTable:
    def __init__(self, counter):
        self.counter = counter

    def bigram(self, verb, dependency, verb2, dependency2):
        """Find all the stories where story contains verb,dependency and verb2,dependency2
        AND they refer to the same entity
        """
        ctr = 0
        for story in self.counter:
            for entity in self.counter[story]:
                v = self.counter[story][entity]
                if (verb, dependency) in v and (verb2, dependency2) in v:
                    ctr +=1
        return ctr

    def unigram(self, verb, dependency):
        """Number of stories containing verb/dependency"""
        ctr = 0
        for story in self.counter:
            for entity in self.counter[story]:
                if (verb, dependency) in self.counter[story][entity]:
                    ctr += 1
        return ctr

    def pmi(self, verb, dependency, verb2, dependency2):
        n = len(self.counter) + PLUS_ONE_SMOOTHING
        prob_a_and_b = self.bigram(verb, dependency, verb2, dependency2)+PLUS_ONE_SMOOTHING/n
        prob_a = self.unigram(verb, dependency)+PLUS_ONE_SMOOTHING/n
        prob_b = self.unigram(verb2, dependency2)+PLUS_ONE_SMOOTHING/n
        return math.log(prob_a_and_b/(prob_a*prob_b))
        #math.log(prob_a_and_b) - (math.log(prob_a) + math.log(prob_b))

    def histo(self, verb, dependency):
        """Return cooccurrence counts for all verb/dependency pairs for a given verb/dependency"""
        ctr = Counter()
        for story in self.counter:
            for entity in self.counter[story]:
                if (verb, dependency) in self.counter[story][entity]:
                    for c in self.counter[story][entity]:
                        if c != (verb, dependency):
                            ctr[c] += 1
        return ctr


    def histo_pmi(self, verb, dependency):
        return sorted([(v, d, self.pmi(verb, dependency, v, d)) for v, d in self.histo(verb, dependency)], key=lambda x: x[-1])


