# http://www.nltk.org/howto/twitter.html

from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random, json

# Normalization: the process of reducing a word to its simplest form
# Example of normalization: ("ran","runs","running" = "run")

# Stemming and Lemmatization are forms of normalization
# Stemming: removes affixes from a word. 
# Lemmatization: similiar to stemming but considers context of the words (part of speech, tense)

# Why normalize the data?
# We want our algorithm to understand these 3 words as one, in order to simplify algorithm and reduce number of 'features' (independent variables)

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    #pos_tag stands for 'part-of-speech tagger'
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

# algorithm's dependent variables (classifications): positive and negative
# algorithm's indendent variables (features): a tweet's text
# create a dictionary with keys -> tokens and values -> True
# (this is a requirement for NLTK's Naive Bayes classifier function) 
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def load_json_into_array(json_filename):
	json_object_list = []
        # each line in the json file contains a seperate json object
	with open(json_filename) as f:
		for line in f.readlines():
			if not line.strip(): # skip empty lines
				continue
			json_object_list.append(json.loads(line))
	return json_object_list

if __name__ == "__main__":
	positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
	negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

	sierra_club_tweet_data_2019 = load_json_into_array('./sierraclub_2019-06_to_2019-12.json')
	sierra_club_tweet_data_2020 = load_json_into_array('./sierraclub_2020-06_to_2020-12.json')

        # tokenize each tweet text
	tokenized_tweets_2019, tokenized_tweets_2020 = [], []
	for tweet in sierra_club_tweet_data_2019:
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

	classifier = NaiveBayesClassifier.train(dataset)

	total_tweets_2019 = len(cleaned_tokens_list_2019)
	total_tweets_2020 = len(cleaned_tokens_list_2020)
	positive_count_2019, positive_count_2020 = 0, 0

	blm_count_2019, blm_count_2020 = 0,0
	blm_positive_2019, blm_positive_2020 = 0,0

	for tweet_tokens in cleaned_tokens_list_2019:
		if "blm" in tweet_tokens:
			blm_count_2019 += 1
		if classifier.classify(dict([token, True] for token in tweet_tokens)) == "Positive":
			positive_count_2019 += 1
			if "blm" in tweet_tokens:
				blm_positive_2019 += 1

	for tweet_tokens in cleaned_tokens_list_2020:
		if "blm" in tweet_tokens:
			blm_count_2020 += 1
		if classifier.classify(dict([token, True] for token in tweet_tokens)) == "Positive":
			positive_count_2020 += 1
			if "blm" in tweet_tokens:
				blm_positive_2020 += 1

	print("Total tweets (06/01/2019 -> 12/01/2019): " + str(total_tweets_2019))
	print("Total tweets (06/01/2020 -> 12/01/2020): " + str(total_tweets_2020))
	print("Total POSITIVE tweets (06/01/2019 -> 12/01/2019): " + str(positive_count_2019))
	print("Total POSITIVE tweets (06/01/2020 -> 12/01/2020): " + str(positive_count_2020))

	print("Positive tweet percentage (06/01/2019 -> 12/01/2019): " + str(positive_count_2019 / total_tweets_2019))
	print("Positive tweet percentage (06/01/2020 -> 12/01/2020): " + str(positive_count_2020 / total_tweets_2020))

	print("Total number of tweets that mention BLM (06/01/2019 -> 12/01/2019): "+ str(blm_count_2019))
	print("Total number of POSITIVE tweets that mention BLM (06/01/2019 -> 12/01/2019): "+ str(blm_positive_2019))
	print("\nTotal number of tweets that mention BLM (06/01/2020 -> 12/01/2020): "+ str(blm_count_2020))
	print("Total number of POSITIVE tweets that mention BLM (06/01/2019 -> 12/01/2019): "+ str(blm_positive_2020))

