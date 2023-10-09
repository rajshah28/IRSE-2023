import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
Data = pd.read_csv(r"IRSE_FIRE_2022_Track_Training_Data_preprocessed.csv",encoding='latin-1')
#X = np.loadtxt(open("IRSE_FIRE_2022_Track_Training_Data_preprocessed.csv", "rb"), delimiter=",", skiprows=1, dtype=str)
#print(Data.shape[0])
# Step - a : Remove blank rows if any.
Data['Comments'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Data['Comments'] = [entry.lower() for entry in Data['Comments']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Data['Comments']= [word_tokenize(entry) for entry in Data['Comments']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Data['Comments']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Data.loc[index,'text_final'] = str(Final_words)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Data['text_final'],Data['Class'],test_size=0.1)
#Train_X = Data['text_final']
#Train_Y = Data['Class']
Encoder = LabelEncoder()
Train_Y1 = Encoder.fit_transform(Train_Y)
Test_Y1 = Encoder.fit_transform(Test_Y)
#print(Train_Y)
#print(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Data['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y1)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y1)*100)
print(precision_recall_fscore_support(predictions_NB,Test_Y1,average='macro'))
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVMc = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVMc.fit(Train_X_Tfidf,Train_Y1)
# predict the labels on validation dataset
predictions_SVM = SVMc.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
# print(predictions_SVM)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y1)*100)
print(precision_recall_fscore_support(predictions_SVM,Test_Y1,average='macro'))
# fit the training dataset on the NB classifier
from csv import writer
from csv import reader

# Open the input_file in read mode and output_file in write mode
with open('IRSE_FIRE_2022_Track_Training_Data_preprocessed.csv', 'r') as read_obj, open('Exp1_Secondary_Results_AritraMitra.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    i=0
    for row in csv_reader:
        if(i == 0):
        # Append the default text in the row / list
            row.append("Predicted Class")
        else:
            row.append(predictions_NB[i-1])
        # Add the updated row / list to the output file
        csv_writer.writerow(row)
        i+=1
with open('IRSE_FIRE_2022_Track_Training_Data_preprocessed.csv', 'r') as read_obj, open('Exp2_Secondary_Results_AritraMitra.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    i=0
    for row in csv_reader:
        if(i == 0):
        # Append the default text in the row / list
            row.append("Predicted Class")
        else:
            row.append(predictions_SVM[i-1])
        # Add the updated row / list to the output file
        csv_writer.writerow(row)
        i+=1
