import streamlit as st
import pandas as pd
import time
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('./ml-25m/movies.csv')


def clean_title(title):
    return re.sub('[^a-zA-Z0-9 ]', '', title)
    
movies['cleaned_title'] = movies['title'].apply(clean_title)






vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies['cleaned_title'])









def search(title):
    tfidf.shape
    
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
    
    res = rec_percentages.head(10).merge(movies, left_index=True,right_on="movieId")[['title','genres']]
    return res


def main():
    # st.title("Movie Recommender")
    html_temp = """
    <div style="background-color:#434390;padding:10px;border-radius:25px;">
    <h2 style="color:white;text-align:center;"> Movie Recommender </h2>
    <h3 style="color:white;text-align:center;">Find movies that people like you love!</h3>
   
    </div>
    <br></br>
    """


    st.markdown(html_temp,unsafe_allow_html=True)
    
    
# # Initialize chat history
#     if "messages" not in st.session_state:
#      st.session_state.messages = []

# # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
  
    with st.chat_message("assistant"):
            message_placeholder = st.empty()
            assistant_response = 'Hello! Type in a movie you like and I will recommend you other movies that you might like!'
            full_response = ''

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
            
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    if prompt := st.chat_input("For example, Toy Story 1995"):
        # st.session_state.messages.append({"role": "assistant", "content": prompt})
    # Display user message in chat message container
        with st.chat_message("user"):
            
            st.markdown(prompt)
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            response = finding_recs(prompt)
            if response.empty:
                assistantResponse = 'Sorry we dont have enough information about this movie as of now. Try again with a different movie name!'
                fullResponse = ''
                message_placeholder.markdown(assistantResponse)
            else:
                textResponse = ''
                
                assistantResponse = 'Here is a list of movies that you might like and their genres!'
                for chunk in assistantResponse.split():
                    textResponse += chunk + ' '
                    time.sleep(0.05)
                    message_placeholder.markdown(textResponse + '▌')
                time.sleep(0.5)
                st.dataframe(response,hide_index=True)
                # st.session_state.messages.append({"role": "user", "content": response})
   

if __name__=='__main__':
    main()