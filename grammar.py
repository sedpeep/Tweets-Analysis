import pandas as pd
import re

#urdu words
urdu_words = ['yoy','hum' ,'ham', 'faiz sahb','subha','aisi','khabron','k', 'sath' ,'hui',
'hai','baap','baghora','chor','lanat','begairto','pe','ki', 'sheeshay','k',
'jaag','sinfe','waley','usi','ko','rahy','hain','sab','aur','hein','khoon','ka',
'mazal','pakistani','pakistaniyu','pakistaniyo','lafafe','haqeeqi','azaadi',
'mein','bhi','rehne','wala','qoum','ko','bata','raha','hun','hogaya',
'aur','inko','ley','paisay','paisy','day','ray','han','waqai','zaroorat',
'aagg','hoi','hrr','trf','aap','aose','aisi','karein','toh','ho','jai','muqabla',
'kidhr','lagti','mulk','chiz','mazak','mazaq','rakh','diya','ny','shkhs',
'haq','ho','gi','aaye','ga','kesy','gya','naa','dalay','awam','ullu','gal','wad','gai',
'banany','waly','sub','hony','saboot','sub','pehly','hy','yeh','kuttey','walla','wala',
'aur','haal','hoga','yazeed','owlad','reha','kro','fouj','ko','zindabaad','zindabad',
'mily','ga','mai','kala','qanoon','sochlia','hota','yeh','na','hota','khubsurat','lagti'
'khatam','nahi','nahin','log','karty','karte','bezti','ki','qanon','nahin',
'bare','ghar','maa','behn','bhai','bai','apky','baal','hifzo','rakhey',
'chal','hat','khush','rakhe','kaisi','jo','rahe','hai','insan','hotey','tu','humay','pe','yeh'
,'zinda','aaj','bhi','izzat','izat','jhooot','jhoot','sach','parho','wasta','chand','aik','waqt',
'waqat','sari','sarii','ye','log','gareebi','baat','acha','choron','mila','aik','wari','kuch','sharam'
'garmi','behan','hain','sb','paisay','day','ray','aur','hum','inko','ley','aisay','ata','hai','saal','magar','aati']


def filter_urdu_tweets(df, urdu_words):
    # Convert urdu_words list to a set for faster look-up
    urdu_words_set = set(urdu_words)

    # Function to check if a tweet contains Urdu words
    def contains_urdu_word(tweet):
        if not isinstance(tweet, str):
            return False
        # Split the tweet into words using regex to get only words
        words = re.findall(r'\b\w+\b', tweet.lower())
        return any(word in urdu_words_set for word in words)

    # Create a boolean mask for tweets containing Urdu words
    mask = df['tweetText'].apply(contains_urdu_word)

    # Extract the rows with tweets containing Urdu words to a new dataframe
    extracted_df = df[mask].copy()

    # Remove the tweets containing Urdu words from the original dataframe
    df = df[~mask].reset_index(drop=True)

    return df, extracted_df

df=pd.read_excel('C:/Users/Omer Habib/PycharmProjects/X_data/cleaned_dataset_without_lemming.xlsx')
original_df, urdu_df = filter_urdu_tweets(df, urdu_words)

print("Original DataFrame (Without Urdu Tweets):")
print(len(original_df))

print("\nExtracted DataFrame (Only Urdu Tweets):")
print(len(urdu_df))
print(urdu_df.head(10))

original_df.to_excel("cleaned_without_roman(without lemming).xlsx")
urdu_df.to_excel("cleaned_with_roman(without lemming).xlsx")