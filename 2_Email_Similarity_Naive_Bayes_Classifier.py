from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
#dataset
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
email_attributes = dir(emails)
target_name = emails.target_names
target = emails.target
#model_train_test
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], 
                                subset='train',
                                shuffle = True, 
                                random_state = 108)

test_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], 
                                 subset='test', 
                                 shuffle=True, 
                                 random_state=108)
#counting words
counter = CountVectorizer()
#fit_transform
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)
#MultinomialNB
classifier = MultinomialNB()
#fit - test_set, test_label
classifier.fit(train_counts, train_emails.target)
#score
score = classifier.score(test_counts, test_emails.target)

print(f" Categories: {np.unique(target_name)}")
print(f"Corresponding labels: {np.unique(emails.target)}")
print(f" Naive Bayes Model Score: {score}")
