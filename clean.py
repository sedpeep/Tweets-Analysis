import pandas as pd
import nltk
import string
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import re



#Download nltk data
nltk.download('words')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')




#initialize
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
english_words = set(words.words())


emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  
    "]+"
)

def clean_dataframe(df):

    #convert column in str
    df['tweetText'] = df['tweetText'].astype(str)
    # Drop rows with missing 'tweetText'
    df = df.dropna(subset=['tweetText'])

    # Remove Whitespace and drop duplicates
    df['tweetText'] = df['tweetText'].str.strip()
    df.drop_duplicates(subset='tweetText', inplace=True)
    print(f"Rows after removing duplicates: {len(df)}")

    # Convert into lowercase
    df['tweetText'] = df['tweetText'].str.lower()

    # Remove URLs
    df['tweetText'] = df['tweetText'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)

    # Remove hashtags
    df['tweetText'] = df['tweetText'].str.replace(r'#', '', regex=True)


    # Replace all @usernames with an empty string
    df['tweetText'] = df['tweetText'].str.replace(r'@\w+', '', regex=True)

    # convert column in str
    df['tweetText'] = df['tweetText'].astype(str)

    # Remove special characters
    df['tweetText'] = df['tweetText'].str.replace(f"[{string.punctuation}]", '', regex=True)

    # Remove emojis
    df['tweetText'] = df['tweetText'].str.replace(emoji_pattern, '', regex=True)


    # Remove stop words and repeated words
    df['tweetText'] = df['tweetText'].apply(remove_stopwords)
    df['tweetText'] = df['tweetText'].apply(remove_repeated_words)

    # Stemming and lemmatization
    # df['stemmedText'] = df['tweetText'].apply(stem_text)
    # df['lemmedText'] = df['tweetText'].apply(lemmatize_sentence)

    # df, df_with_words = filter_tweets(df)
    # df_with_words.to_excel('filtered.xlsx')
    # # Sentiment Analysis
    df['Sentiment'] = df['tweetText'].apply(analyze_sentiment)

    df = df.dropna(subset=['tweetText'])

    return df


#urdu words
words = ['yoy','hum' ,'ham', 'faiz sahb','subha','aisi','khabron','k', 'sath' ,'hui',
'hai','baap','baghora','chor','lanat','begairto','pe','ki', 'sheeshay','k',
'jaag','sinfe','waley','usi','ko','rahy','hain','sab','aur','hein','khoon','ka',
'mazal','pakistani','pakistaniyu','pakistaniyo','lafafe','haqeeqi','azaadi',
'mein','bhi','rehne','wala','qoum','ko','bata','raha','hun','hogaya',
'aur','inko','ley','paisay','paisy','day','ray','han','waqai','zaroorat',
'aagg','hoi','hrr','trf','aap','aose','karein','toh','ho','jai','muqabla',
'kidhr','lagti','mulk','chiz','mazak','mazaq','rakh','diya','ny','shkhs',
'haq','ho','gi','aaye','ga','kesy','gya','naa','dalay','awam','ullu','gal','wad','gai',
'banany','waly','sub','hony','saboot','sub','pehly','hy','yeh','kuttey','walla','wala',
'aur','haal','hoga','yazeed','owlad','reha','kro','fouj','ko','zindabaad','zindabad',
'mily','ga','mai','kala','qanoon','sochlia','hota','yeh','na','hota',
'khatam','nahi','nahin','log','karty','karte','bezti','ki','qanon','nahin',
'bare','ghar','maa','behn','bhai','bai','apky','baal','hifzo','rakhey',
'chal','hat','khush','rakhe','kaisi','jo','rahe','hai','insan','hotey','tu','humay','pe','yeh'
,'zinda','aaj','bhi','izzat','izat','jhooot','jhoot','sach','parho','wasta','chand','aik','waqt',
'waqat','sari','sarii','ye','log','gareebi','baat','acha','choron','mila','aik','wari','kuch','sharam'
'garmi','behan','hain','sb']


def filter_tweets(df, column_name='tweetText'):
    # Filtering rows that contain any of the words in word_list
    contains_word_df = df[df[column_name].str.contains('|'.join(words), case=False, na=False)]

    # Rows that do not contain the words
    does_not_contain_word_df = df[~df.index.isin(contains_word_df.index)]

    return does_not_contain_word_df, contains_word_df


# def extract_roman_tweets(df, column_name):
#     # Counter for tweets checked
#     tweets_checked = 0
#
#     # Filter function
#     def is_likely_english(tweet):
#         nonlocal tweets_checked
#         print(tweets_checked)
#         tweets_checked += 1
#         try:
#             # If the detected language isn't English, it's not a Roman tweet
#             if detect(tweet) != 'en':
#                 return True
#             # If the tweet is detected as English, check individual words
#             tokens = tweet.split()
#             english_word_count = sum(1 for token in tokens if is_english_word(token))
#             # If more than 70% of the words are English, consider it an English tweet
#             return english_word_count / len(tokens) > 0.7
#         except:
#             return True  # In case of any error, assume it's English
#
#     roman_tweets_df = df[~df[column_name].apply(is_likely_english)]
#     print(roman_tweets_df)
#     print(f"Total tweets checked: {tweets_checked}")
#
#     return roman_tweets_df

#Remove repeated words like 'very'
def remove_repeated_words(text):
    previous_word = ''
    cleaned_text = []
    for word in text.split():
        if word != previous_word:
            cleaned_text.append(word)
        previous_word = word
    return ' '.join(cleaned_text)

#Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to map NLTK's POS tag to first character used by WordNetLemmatizer
def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatization function for sentences
def lemmatize_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
    return ' '.join(lemmatized_words)

# Stemming function
def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

#remove sstop words
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

if __name__ == "__main__":
    df = pd.read_excel('C:/Users/Omer Habib/PycharmProjects/X_data/translated_urdu_tweets.xlsx')
    print(f"Original number of rows: {len(df)}")
    cleaned_df = clean_dataframe(df)
    cleaned_df.to_excel("cleaned_urdu_translated_dataset_without_lemming.xlsx", index=False)


