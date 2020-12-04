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
import re, string

# normalization is the porcess of reducing a word to its simplest form
# example of normalization: ("ran","runs","running" = "run")

# stemming and lemmatization are forms of normalization
# Stemming: removes affixes from a word. 
# Lemmatization: similiar to stemming but considers context of the words (part of speech, tense)

# Why normalize the data?
# we want our algorithm to understand these 3 words as one, in order to simplify algorithm and reduce number of 'features' (independent variables)

# algorithm's dependent variables (classifiers): positive and negative
# algorithm's indendent variables (features): a tweet's text
 
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

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

print("\nPositive tweet (untokenized):")
print(positive_tweets[0])
print("\nPositive tweet (tokenized / part-of-speech tagged):")
#note: pos_tag stands for 'part-of-speech tagger'
# part of speech: "In English the main parts of speech are noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection."

print(pos_tag(tweet_tokens[0]))

print("\nPositive tweet (lemmatized / noise removed):")
# a stop word is the kind of word a search engine ignores to be efficient (is, at, my, I, be, etc.)
stop_words = stopwords.words('english')
print(remove_noise(tweet_tokens[0], stop_words))

