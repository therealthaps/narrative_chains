import chains

data, table = chains.process_corpus("train.csv", 100)

for dat in data:
    print(dat)
    print()
    print()
print(table.unigram("move", "nsubj"))

# 1 Testing
# 2 load the testing data.
#       using panda 
# 3 read first four sentences of 5 sentence narrative chain and extract their (verb, dependency) using spacy
#       can be done same chains.py 
# 4 read Two different options for the last sentence and extract their dependencies too using spacy
#       final sentencces and their dependencies using same process as in chains.py
# 5 find all the stories in testing set which has the same verb, dependency set
#       we can search the data we get from chains.process_corpus and find the similar verb dependency of the story.
# 6 calculate the pmi of each of verb dependency and with the verb dependecies of two possible final sentences.
#       calculate pmi using table.pmi(a,b,c,d)
# 7 each of the pmi of verb dependency calculated in the above step should be higer than the pmi of the verbs alone
#       
# 8 Finally, between those two possible verb,dependency set, the one with the higher pmi value should be the correct ending of the story according to model.
#       compare the value and select final sentences.

# 9 Check if the guess made by our model is correct.
# 10 all the stories that were found in step 5 should be considered.
# 11 We see which story/stories have the similar/same verb, dependency set and we check if the verb dependency set we selected for the last sentences matches the one in that story. 