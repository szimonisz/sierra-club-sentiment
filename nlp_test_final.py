# http://www.nltk.org/howto/twitter.html
# importing modules (python-term) from the nltk library

from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re, string, random
import json

# normalization is the process of reducing a word to its simplest form
# example of normalization: ("ran","runs","running" = "run")

# stemming and lemmatization are forms of normalization
# Stemming: removes affixes from a word. 
# Lemmatization: similiar to stemming but considers context of the words (part of speech, tense)

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

    #note: pos_tag stands for 'part-of-speech tagger'
    # part of speech: "In English the main parts of speech are noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection."
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


if __name__ == "__main__":
	positive_tweets = twitter_samples.strings('positive_tweets.json')
	negative_tweets = twitter_samples.strings('negative_tweets.json')
	positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
	negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

	sierra_club_tweet_data_2019, sierra_club_tweet_data_2020 = [], []
        # each line in the json file contains a seperate json object
	with open('./sierraclub_2019-06_to_2019-12.json') as f:
		for line in f.readlines():
			if not line.strip(): # skip empty lines
				continue
			sierra_club_tweet_data_2019.append(json.loads(line))
	with open('sierraclub_2020-06_to_2020-12.json') as f:
		for line in f.readlines():
			if not line.strip(): # skip empty lines
				continue
			sierra_club_tweet_data_2020.append(json.loads(line))

	tokenized_tweets_2019, tokenized_tweets_2020 = [], []
	for tweet in sierra_club_tweet_data_2019:
		#print(tweet)
		tweet_text = tweet['tweet']
		tokenized_tweets_2019.append(remove_noise(word_tokenize(tweet_text)))

	for tweet in sierra_club_tweet_data_2020:
		tweet_text = tweet['tweet']
		tokenized_tweets_2020.append(remove_noise(word_tokenize(tweet_text)))

	# a stop word is the kind of word a search engine ignores to be efficient (is, at, my, I, be, etc.)
	stop_words = stopwords.words('english')

	positive_cleaned_tokens_list, negative_cleaned_tokens_list = [], []
	for tokens in positive_tweet_tokens:
	    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
	for tokens in negative_tweet_tokens:
	    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

	#all_positive_words = get_all_words(positive_cleaned_tokens_list)
	#freq_dist_positive = FreqDist(all_positive_words)
	#print("\nTop 10 Most common words found in all positive tweets")
	#print(freq_dist_positive.most_common(10))

	cleaned_tokens_list_2019, cleaned_tokens_list_2020 = [], []
	for tokens in tokenized_tweets_2019:
	    cleaned_tokens_list_2019.append(remove_noise(tokens, stop_words))
	for tokens in tokenized_tweets_2020:
	    cleaned_tokens_list_2020.append(remove_noise(tokens, stop_words))

	positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
	negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

	positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
	negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

	dataset = positive_dataset + negative_dataset
	random.shuffle(dataset)
	train_data = dataset[:7000]
	test_data = dataset[7000:]

	classifier = NaiveBayesClassifier.train(train_data)

	total_tweets_2019 = len(cleaned_tokens_list_2019)
	total_tweets_2020 = len(cleaned_tokens_list_2020)
	print("Total tweets 06/01/2019 -> 12/01/2019: " + str(total_tweets_2019))
	print("Total tweets 06/01/2020 -> 12/01/2020: " + str(total_tweets_2020))
	positive_count_2019, positive_count_2020 = 0, 0

	for tweet_tokens in cleaned_tokens_list_2019:
		if classifier.classify(dict([token, True] for token in tweet_tokens)) == "Positive":
			positive_count_2019 += 1
	for tweet_tokens in cleaned_tokens_list_2020:
		if classifier.classify(dict([token, True] for token in tweet_tokens)) == "Positive":
			positive_count_2020 += 1
	print("Positive tweet percentage in 2019: ")
	print(positive_count_2019 / total_tweets_2019)
	print("Positive tweet percentage in 2020: ")
	print(positive_count_2020 / total_tweets_2020)
