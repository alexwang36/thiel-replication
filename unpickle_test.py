import pickle as pkl

file_name = 'lda_5.pkl'
file = open(file_name, 'rb')
lda_5 = pkl.load(file)
file.close()

topics = lda_5.components_
shape = lda_5.components_.shape

print("type of loaded object: ", type(lda_5))
print("attributes and methods of loaded object: ", dir(lda_5))
print("topics: ", topics)
print("shape of topic-term distribution: ", shape)

