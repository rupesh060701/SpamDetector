import streamlit as st
import pickle   
import string

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def transform_text(text):
  text=text.lower()
  text = nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
          y.append(i)

  text = y[:]
  y.clear()

  for i in text:
      y.append(ps.stem(i))
  return " ".join(y)

from flask import Flask
app = Flask(__name__)



st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter the message")

if st.button('Predict'):

    transformed_sms=transform_text(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not spam")




# if __name__ == "__main__":
#     app.run(debug=True) 

# if you want to change port use port=8000  in app.run()

