import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

song = ""

@st.cache(allow_output_mutation=True)
class FaceEmotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                song = output
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

#------------------------------------------------------------------------------------------------------------------------------------___#


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("data/filtered_track_df.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios
 
def page():
    
    st.markdown("##")

    with st.container():
        col1, col2,col3,col4 = st.columns((2,0.5,0.5,0.5))
        with col3:
            st.markdown("***Choose your genre:***")
            genre = st.radio(
                "",
                genre_names, index=genre_names.index("Pop"))
        with col1:
            st.markdown("***Choose features to customize:***")
            start_year, end_year = st.slider(
                'Select the year range',
                1990, 2019, (2015, 2019)
            )
            acousticness = st.slider(
                'Acousticness',
                0.0, 1.0, 0.5)
            danceability = st.slider(
                'Danceability',
                0.0, 1.0, 0.5)
            energy = st.slider(
                'Energy',
                0.0, 1.0, 0.5)
            instrumentalness = st.slider(
                'Instrumentalness',
                0.0, 1.0, 0.0)
            valence = st.slider(
                'Valence',
                0.0, 1.0, 0.45)
            tempo = st.slider(
                'Tempo',
                0.0, 244.0, 118.0)

    tracks_per_page = 6
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
    
    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to recommend")


def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection and Song Recomendation Application")
    activiteis = ["Home", "Realtime Detection", "Data"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by:
                 \nAI-Summer Semester Class
            \nInstructor: 
            \nSir Shahzeb Khan
            """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 3. Song Recommendation based on Emotion

                 """)
    elif choice == "Realtime Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=FaceEmotion)

        html_temp_song = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Song Recommendation.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_song, unsafe_allow_html=True)

        page()

        # if (song == "neutral"):
        #     st.write("Since you mood seems NEUTRAL I would recommend the song: ")
        # if (song == "happy"):
        #     st.write("Since you mood seems HAPPY I would recommend the song: ")
        # if (song == "sad"):
        #     st.write("Since you mood seems SAD I would recommend the song: ")
        # if (song == "angry"):
        #     st.write("Since you mood seems ANGRY I would recommend the song: ")
        # if (song == "surprise"):
        #     st.write("Since you mood seems SURPRISE I would recommend the song: ")

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()