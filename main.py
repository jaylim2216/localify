import os

from flask import Flask, session, redirect, url_for, request, render_template
import pandas as pd
import numpy as np
import json
import re
import sys
import itertools
import time


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

client_id = 'f9f751d90ac24930b24521beeb0b95b1'
client_secret = 'c8df92ccd956473c8cc1475f645ad235'
redirect_uri = 'http://jaylim2216.pythonanywhere.com/callback'
scope = 'user-library-read playlist-read-private playlist-modify-public playlist-modify-private'

cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog = True
)
sp = Spotify(auth_manager=sp_oauth)

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('choose_playlist'))

@app.route('/callback')
def callback():
    sp_oauth.get_access_token(request.args['code'])
    return redirect(url_for('choose_playlist'))

@app.route('/choose_playlist', methods=['GET', 'POST'])
def choose_playlist():
    if request.method == 'POST':
        playlist_name = request.form['playlist_name']
        print(f"Received playlist name: {playlist_name}")

        # Fetch user's playlists to validate input
        id_name = {}
        list_photo = {}
        for i in sp.current_user_playlists()['items']:
            id_name[i['name']] = i['uri'].split(':')[2]
            list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']

        print(f"Available playlists: {id_name.keys()}")

        if playlist_name not in id_name:
            print(f"Playlist '{playlist_name}' not found.")
            return f"Playlist '{playlist_name}' not found. Please choose a valid playlist. Press Back to try again."

        session['playlist_name'] = playlist_name
        session['id_name'] = id_name
        print("Redirecting to get_recommendations")
        return redirect(url_for('get_recommendations'))

    playlists = sp.current_user_playlists()['items']
    print(f"Playlists: {playlists}")  # Debug print
    return render_template('playlist_choice.html', playlists=playlists)

def log_dataframe_to_file(df, filename):
    with open(filename, 'w') as file:
        file.write(df.to_string())

@app.route('/get_recommendations', methods = ['GET', 'POST'])
def get_recommendations():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)

    playlist_choice = session.get('playlist_name', None)
    id_name = session.get('id_name', None)

    if playlist_choice is None or id_name is None:
        return redirect(url_for('choose_playlist'))

    # Step 1: Load and preprocess the data
    def explore_and_prepare_data(file_path):
        spotify_df = pd.read_csv(file_path)
        spotify_df['Genres'] = spotify_df['Genres'].replace('nan', '').astype(str)
        spotify_df['Genres'] = spotify_df['Genres'].replace('nan', '').astype(str)
        spotify_df['Genres_upd'] = spotify_df['Genres'].apply(lambda x: re.findall(r"[^,]+", x))
        spotify_df['Artists_upd'] = spotify_df['Artist Name(s)'].apply(lambda x: re.findall(r"[^,]+", x))
        spotify_df['year'] = spotify_df['Added At'].apply(lambda x: x.split('-')[0])
        spotify_df['popularity_red'] = spotify_df['Popularity'].apply(lambda x: int(x/5))
        return spotify_df

    #simple function to create OHE features
    def ohe_prep(df, column, new_name):
        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)
        return tf_df

    # Step 2: Create feature set
    def create_feature_set(df, float_cols, feature_weights=None):
        print("Genres_upd column content before processing:")
        print(df['Genres_upd'].head(10))
        # Verify the contents of the 'Genres_upd' column
        if df['Genres_upd'].apply(lambda x: " ".join(x)).str.strip().eq('').all():
            raise ValueError("Genres_upd column is empty or contains only stop words.")

        # tfidf genre lists
        tfidf = TfidfVectorizer()  # Include stop words removal
        tfidf_matrix = tfidf.fit_transform(df['Genres_upd'].apply(lambda x: " ".join(x)))

        # Check if tfidf_matrix is empty
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("TF-IDF matrix is empty. Check the contents of 'Genres_upd' column.")

        genre_df = pd.DataFrame(tfidf_matrix.toarray())
        genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
        genre_df.reset_index(drop=True, inplace=True)

        # other OHE features
        year_ohe = ohe_prep(df, 'year', 'year') * 0.5
        popularity_ohe = ohe_prep(df, 'popularity_red', 'pop') * 0.15

        # scale float columns
        floats = df[float_cols].reset_index(drop=True)
        scaler = MinMaxScaler()
        floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns)* 0.1

        if feature_weights:
            for feature, weight in feature_weights.items():
                if feature in floats_scaled.columns:
                    floats_scaled[feature] *= weight
                elif feature in genre_df.columns:
                    genre_df[feature] *= weight
                elif feature in year_ohe.columns:
                    year_ohe[feature] *= weight
                elif feature in popularity_ohe.columns:
                    popularity_ohe[feature] *= weight

        # concatenate all features
        final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)

        # add song id
        final['id'] = df['Spotify ID'].values

        return final


    def create_necessary_outputs(playlist_name,id_dic, df):
        #generate playlist dataframe
        playlist = pd.DataFrame()
        playlist_name = playlist_name

        for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
            #print(i['track']['artists'][0]['name'])
            playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
            playlist.loc[ix, 'Track Name'] = i['track']['name']
            playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
            playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
            #playlist.loc[ix, 'date_added'] = i['added_at']
            playlist.loc[ix, 'date_added'] = pd.Timestamp.now()
        playlist['date_added'] = pd.to_datetime(playlist['date_added'])


        #playlist = playlist[playlist['id'].isin(df['Spotify ID'].values)].sort_values('date_added',ascending = False)
        playlist = playlist.sort_values('date_added', ascending=False)
        return playlist


    def fetch_song_features_from_spotify(sp, song_ids):
        """Fetch song features from Spotify API for given song IDs."""
        features_list = []

        for song_id in song_ids:
            # Fetch audio features
            audio_features = sp.audio_features(song_id)[0]

            if audio_features:
                # Fetch track details
                track = sp.track(song_id)
                album = track['album']
                artists = track['artists']
                artist_ids = [artist['id'] for artist in artists]
                artist_names = [artist['name'] for artist in artists]

                # Fetch artist genres
                genres = []
                for artist_id in artist_ids:
                    artist_info = sp.artist(artist_id)
                    genres.extend(artist_info['genres'])

                # Prepare the data for the DataFrame
                features = {
                    'Spotify ID': track['id'],
                    'Artist IDs': ",".join(artist_ids),
                    'Track Name': track['name'],
                    'Album Name': album['name'],
                    'Artist Name(s)': ",".join(artist_names),
                    'Release Date': album['release_date'],
                    'Duration (ms)': track['duration_ms'],
                    'Popularity': track['popularity'],
                    'Added By': 'spotify:user:bandwagonapp',  # Placeholder
                    'Added At': '2024-05-17T05:43:10Z',  # Placeholder
                    'Genres': ",".join(genres),
                    'Danceability': audio_features['danceability'],
                    'Energy': audio_features['energy'],
                    'Key': audio_features['key'],
                    'Loudness': audio_features['loudness'],
                    'Mode': audio_features['mode'],
                    'Speechiness': audio_features['speechiness'],
                    'Acousticness': audio_features['acousticness'],
                    'Instrumentalness': audio_features['instrumentalness'],
                    'Liveness': audio_features['liveness'],
                    'Valence': audio_features['valence'],
                    'Tempo': audio_features['tempo'],
                    'Time Signature': audio_features['time_signature'],
                    'Genres_upd': genres,
                    'Artists_upd': artist_names,
                    'year': pd.to_datetime(album['release_date']).year,
                    'popularity_red': int(track['popularity'] / 10)  # Simplified popularity_red calculation
                }
                features_list.append(features)

        return pd.DataFrame(features_list)
    def generate_playlist_feature(playlist_df, complete_feature_set, weight_factor):
        """
        Summarize a user's playlist into a single vector

        Parameters:
            playlist_df (pandas dataframe): playlist dataframe
            complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
            weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1.

        Returns:
            playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        """

        print("Generating playlist feature set...")

        # Fetch features for songs in the playlist directly from Spotify
        playlist_features_df = fetch_song_features_from_spotify(sp, playlist_df['id'].values)
        print(f"Fetched features for playlist songs:\n{playlist_features_df.head(10)}")

        # Ensure consistency with complete feature set columns
        complete_feature_cols = complete_feature_set.columns.drop('id')

        # Apply the same feature transformations to playlist features
        important_features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
        float_cols = [col for col in complete_feature_set.columns if col in important_features]

        # Create feature set for playlist features
        playlist_feature_set = create_feature_set(playlist_features_df, float_cols)

        # Ensure all necessary columns are present in playlist features
        for col in complete_feature_cols:
            if col not in playlist_feature_set.columns:
                playlist_feature_set[col] = 0.0

        # Merge with playlist_df to get date_added
        playlist_feature_set = playlist_feature_set.merge(playlist_df[['id', 'date_added']], on='id', how='inner')

        # Sort the playlist features by date_added
        playlist_feature_set = playlist_feature_set.sort_values('date_added', ascending=False)

        # Calculate the most recent date
        most_recent_date = playlist_feature_set.iloc[0, -1]

        # Calculate months from the most recent date for recency bias
        for ix, row in playlist_feature_set.iterrows():
            playlist_feature_set.loc[ix, 'months_from_recent'] = int((most_recent_date.to_pydatetime() - row['date_added'].to_pydatetime()).days / 30)

        # Apply weight based on recency
        playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))

        # Apply weights to the features
        playlist_feature_set_weighted = playlist_feature_set.copy()
        playlist_feature_set_weighted.update(playlist_feature_set_weighted[complete_feature_cols].mul(playlist_feature_set_weighted['weight'], 0))
        playlist_feature_set_weighted_final = playlist_feature_set_weighted[complete_feature_cols]

        print("Playlist feature set weighted final:")
        print(playlist_feature_set_weighted_final.head(10))

        return playlist_feature_set_weighted_final.sum(axis=0)





    # Step 3: Generate recommendations
    def generate_playlist_recos(df, features, nonplaylist_features):
        """
        Pull songs from a specific playlist.

        Parameters:
            df (pandas dataframe): spotify dataframe
            features (pandas series): summarized playlist feature
            nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

        Returns:
            non_playlist_df_top_40: Top 40 recommendations for that playlist
        """

        print("Generating recommendations...")
        print(f"Features vector:\n{features}")

        # Ensure no NaN values in the feature sets
        nonplaylist_features = nonplaylist_features.fillna(0)
        features = features.fillna(0)

        # Ensure feature consistency
        for col in nonplaylist_features.columns.drop('id'):
            if col not in features.index:
                features[col] = 0.0

        # Filter non-playlist features
        non_playlist_df = df[df['Spotify ID'].isin(nonplaylist_features['id'].values)]

        # Calculate cosine similarity
        print(f"Non-playlist features shape: {nonplaylist_features.shape}")
        print(f"Features shape: {features.shape}")
        non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis=1).values, features.values.reshape(1, -1))[:, 0]

        # Get top 40 recommendations based on similarity
        non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40)

        # Add the album image URL to the recommendations
        non_playlist_df_top_40['url'] = non_playlist_df_top_40['Spotify ID'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])

        print("Top 40 Recommendations:\n", non_playlist_df_top_40[['Spotify ID', 'Track Name', 'sim']].head(10))

        return non_playlist_df_top_40

    feature_weights = {
        'Danceability': 1.2,
        'Energy': 1.2,
        'Valence': 1.2,
        'Tempo': 0.8,
        'Loudness': 0.8,
        'Speechiness': 0.5,
        'Acousticness': 1.2,
        'Instrumentalness': 1.2
       # Add weights for genre columns as needed
    }


    # Load and process data
    spotify_df = explore_and_prepare_data('/home/jaylim2216/mysite/sgmusic.csv')
    float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
    complete_feature_set = create_feature_set(spotify_df, float_cols, feature_weights)

    playlist = create_necessary_outputs(playlist_choice ,id_name, spotify_df)
    complete_feature_set_playlist_vector = generate_playlist_feature(playlist, complete_feature_set, 1.15)
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist['id'].values)]
    recc_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)
    print(f"Top 40 Recommendations: {recc_top40.head(10)}")
    recc_top10 = recc_top40.head(10)
    recc_top10['spotify_url'] = recc_top10['Spotify ID'].apply(lambda x: f"https://open.spotify.com/track/{x}")
    temp_file = 'top_40_recommendations.csv'
    recc_top40.to_csv(temp_file, index=False)

    session['top_40_playlist'] = temp_file
    #session['top_40_playlist'] = recc_top40.to_dict(orient='records')

    return render_template('top_10_recommendations.html', recommendations=recc_top10.to_dict(orient='records'))


@app.route('/create_top_40_playlist', methods=['POST'])
def create_top_40_playlist():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)

    # Step 4: Create a Spotify playlist and add tracks
    def create_spotify_playlist(sp, user_id, playlist_name, playlist_description):
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=playlist_description)
        return playlist['id']

    def add_tracks_to_playlist(sp, playlist_id, track_ids):
        sp.playlist_add_items(playlist_id, track_ids)

    user_id = sp.me()['id']
    playlist_name = f"Your Local Music Mix for {session['playlist_name']}"
    playlist_description = f"Top 40 recommended local songs based on {session['playlist_name']}"
    #top_40_recommendations = session.get('top_40_playlist')

    #if not top_40_recommendations:
        #return redirect(url_for('choose_playlist'))
    temp_file = session.get('top_40_playlist', None)
    if temp_file and os.path.exists(temp_file):
        top_40_recommendations = top_40_recommendations = pd.read_csv(temp_file).to_dict(orient='records')
    else:
        return redirect(url_for('choose_playlist'))

    if not top_40_recommendations:
        return redirect(url_for('choose_playlist'))
    print(f"Top 40 Recommendations from session: {top_40_recommendations}")
    playlist_id = create_spotify_playlist(sp, user_id, playlist_name, playlist_description)
    add_tracks_to_playlist(sp, playlist_id, [track['Spotify ID'] for track in top_40_recommendations])
    session['playlist_id'] = playlist_id

    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    playlist_name = session.get('playlist_name', 'your playlist')
    playlist_id = session.get('playlist_id', '')
    print(f"Playlist ID: {playlist_id}")
    print(f"Playlist Name: {playlist_name}")
    return render_template('thank_you.html', playlist_name=playlist_name, playlist_id = playlist_id)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Your existing functions (explore_and_prepare_data, ohe_prep, create_feature_set, create_necessary_outputs,
# fetch_song_features_from_spotify, generate_playlist_feature, generate_playlist_recos, create_spotify_playlist,
# add_tracks_to_playlist) should be included here

if __name__ == '__main__':
    app.run(debug=True)
