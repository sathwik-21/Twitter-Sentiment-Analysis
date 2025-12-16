from flask import render_template,request,Flask
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


app=Flask(__name__)

model=load_model("model.keras")

#load tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)

max_len=60
label_map={
    0:'Negative',
    1:'Positive',
    2:'Neutral',
    3:'Irrelevant'
}

#Preprocessing
import string
def clean_text(text):
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    return text


def predict_sentiment(text):
    text=clean_text(text)
    seq=tokenizer.texts_to_sequences([text])
    padded=pad_sequences(seq,maxlen=max_len,padding='post',truncating='post')
    preds=model.predict(padded)
    class_id=preds.argmax(axis=1)[0]
    return label_map[class_id]

@app.route("/",methods=["GET","POST"])
def index():
    prediction=None
    if request.method=="POST":
        tweet=request.form['tweet']
        prediction=predict_sentiment(tweet)
    return render_template("home.html",prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)