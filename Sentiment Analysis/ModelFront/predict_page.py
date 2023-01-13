import streamlit as st
import pickle
import tensorflow as tf

def load_model():
    with open("saved_steps.pkl", "rb") as f:
        data = pickle.load(f)
        return data

data = load_model()
tokenizer = data['tokenizers']
pad_sequences = data['pad_sequence']
sentiment_label = {1: 'Negative', 0: 'Positive'}
model2 = tf.keras.models.load_model('64x3-CNN.model')
def show_predict_page():

    st.title("Sentiment Analysis")

    st.write("Let's predict the sentiment of your text using Neural Network!")

    text = st.text_input('Input your sentence here:')
    ok = st.button("Analyze Sentiment")
    if ok:
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw,maxlen=200)
        prediction = int(model2.predict(tw).round().item())
        if(prediction == 0):
            st.success("""The sentiment of your sentence "{}" is positive""".format(text))
        else:
            st.error("""The sentiment of your sentence "{}" is negative""".format(text))
        
        
        
        
         