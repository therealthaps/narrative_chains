import chains


def parse_test_instance(story):
    """Returns TWO ParsedStory instances representing option 1 and 2"""
    # this is very compressed
    id = story.InputStoryid
    story = list(story)
    sentences = [chains.nlp(sentence) for sentence in story[2:6]]
    alternatives = [story[6], story[7]]
    return [chains.ParsedStory(id, id, chains.nlp(" ".join(story[2:6]+[a])), *(sentences+[chains.nlp(a)])) for a in alternatives]


# Load training data
data, table = chains.process_corpus("train.csv", 100)
print(table.pmi("move", "nsubj", "move", "nsubj"))

# load testing data
test = load_data("test.csv")
for t in test:
    one, two = parse_test_instance(t)
    one_deps = extract_dependency_pairs(one)
    # logic to choose between one and two
