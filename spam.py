import pandas as pd  #handle the data

from sklearn.model_selection import train_test_split   # used to split datast to train data and test data

from sklearn.feature_extraction.text import CountVectorizer  #used to convert text to decimal data

from sklearn.naive_bayes import MultinomialNB  # used to classify the data

import streamlit as st             # to built web application

data = pd.read_csv("C:\project(ml)\spamdetection\spam.csv")

#clean the data

data.drop_duplicates(inplace=True)

#replacing ham to not spam and spam to Spam

data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])


#categorized the dataset

mesg = data['Message']    #input dataset
cat = data['Category']    #output dataset

#split the data into 80% of train data and 20% of test data

(mesg_train,mesg_test,cat_train,cat_test) =train_test_split(mesg, cat, test_size=0.2)

cv =CountVectorizer(stop_words='english')  #words like a,an,the,i will be filtered out bcoz the do not give more context or important meaning

features = cv.fit_transform(mesg_train) #tranform text to numeric format

#creating the model

model = MultinomialNB()

#trainging the model

model.fit(features, cat_train)


#testing the model

features_test = cv.transform(mesg_test)  # inputdataset into numerical format

#predict the data

#create a function to input from the user 

def predict(message):
    input_message = cv.transform([message]).toarray()  #real_time_data to predict and converting it to array
    result = model.predict(input_message)
    return result

#built web application 

st.header('Spam Dectection')

input_mesg = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mesg)
    #st.markdown(output)
    st.markdown(f"**Prediction:** {output[0]}")