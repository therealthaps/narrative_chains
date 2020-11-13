import chains

data, table = chains.process_corpus("train.csv", 100)

print(table.pmi("move", "nsubj", "move", "nsubj"))
