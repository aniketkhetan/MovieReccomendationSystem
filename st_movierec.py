import streamlit as st
import pandas as pd
import sklearn as sk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle 

movies = pd.read_csv('./ml-25m/movies.csv')


def clean_title(title):
    return re.sub('[^a-zA-Z0-9 ]', '', title)
    
movies['cleaned_title'] = movies['title'].apply(clean_title)






vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies['cleaned_title'])
tfidf = tfidf.shape







def search(title):
  
    title = clean_title(title=title)
    query_vector = vectorizer.transform([title])
    similarity = cosine_similarity(query_vector,tfidf).flatten()
    indices = np.argpartition(similarity,-5)[-5:]
    result = movies.iloc[indices][::-1]
    return result





ratings = pd.read_csv('./ml-25m/ratings.csv')


def finding_recs(title):
   
    res = search(title)
    movie_Id = res.iloc[0]['movieId']
    similar_users = ratings[(ratings['movieId'] == movie_Id) & (ratings['rating'] > 4)]['userId'].unique()
    similar_user_rec = ratings[(ratings['userId'].isin(similar_users)) & (ratings['rating']> 4)]['movieId']
    similar_user_rec = similar_user_rec.value_counts()/len(similar_users)
    similar_user_rec = similar_user_rec[similar_user_rec > 0.1]
    all_users = ratings[(ratings["movieId"].isin(similar_user_rec.index)) & (ratings["rating"] > 4)]

    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_rec, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages['score'] = rec_percentages['similar']/rec_percentages['all']
    rec_percentages = rec_percentages.sort_values('score',ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[['score','title','genres']]
    


def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    movietitle  = st.text_input("Movie","Type Here")
  
    
    if st.button("Predict"):
        results = finding_recs(movietitle)
        st.dataframe(results)
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()