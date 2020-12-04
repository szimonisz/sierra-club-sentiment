# http://www.nltk.org/howto/twitter.html
# importing modules (python-term) from the nltk library
# an instance of a class is an 'object'
# a method is called onto an object -- usually to alter its state.
# a method is implicity used for an object for which it is called.
# the method is accessible to data that is contained within the class.
# example syntax:
# class ClassName:
#     def method_name():
#          # method body
# a function is a function

from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re, string, random

# normalization is the porcess of reducing a word to its simplest form
# example of normalization: ("ran","runs","running" = "run")

# stemming and lemmatization are forms of normalization
# Stemming: removes affixes from a word. 
# Lemmatization: similiar to stemming but considers context of the words (part of speech, tense)
# lemmatization considers part of speech (pos)

# Why normalize the data?
# we want our algorithm to understand these 3 words as one, in order to simplify algorithm and reduce number of 'features' (independent variables)

# algorithm's dependent variables (classifiers): positive and negative
# algorithm's indendent variables (features): a tweet's text

# create dictionary with keys -> tokens and values -> True
# (this is a requirement for NLTK's Naive Bayes classifier function) 

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        # if token is a hyperlink, replace it with an empty string
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        # if token is a twitter handle, set token as an empty string
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token,pos)
         
        # if token is not an empty string, punctuation mark(s), or stop word, keep it
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            #convert word to lower case
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# twitter_samples is a collection of 20k sample tweets stored as line-seperated JSON
# the strings() method of twitter_samples will print all of the tweets as strings

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')


print("\nPositive tweet (untokenized):")
print(positive_tweets[0])
print("\nPositive tweet (tokenized / part-of-speech tagged):")
#note: pos_tag stands for 'part-of-speech tagger'
# part of speech: "In English the main parts of speech are noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection."

print(pos_tag(positive_tweet_tokens[0]))

print("\nPositive tweet (lemmatized / noise removed):")
# a stop word is the kind of word a search engine ignores to be efficient (is, at, my, I, be, etc.)
stop_words = stopwords.words('english')
print(remove_noise(positive_tweet_tokens[0], stop_words))

# an array of arrays
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

print("\n 500th positive tweet vs cleaned 500th positive tweet:")
print(positive_tweet_tokens[500])
print(positive_cleaned_tokens_list[500])

all_positive_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_positive = FreqDist(all_positive_words)

print("\nTop 10 Most common words found in all positive tweets")
print(freq_dist_positive.most_common(10))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

custom_tweet = "This place sucks! I can't believe how rude these people are."
custom_tokens = remove_noise(word_tokenize(custom_tweet))
print("\nCustom tweet: " + custom_tweet)
print(classifier.classify(dict([token, True] for token in custom_tokens)))
custom_tweet2 = "This store is amazing! Best place in town."
custom_tokens2 = remove_noise(word_tokenize(custom_tweet2))
print("\nCustom tweet: " + custom_tweet2)
print(classifier.classify(dict([token, True] for token in custom_tokens2)))
