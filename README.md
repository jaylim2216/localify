## Localify: Singaporean Music Recommendation Algorithm with Spotify

Welcome to Localify! This is a project I started from scratch in Summer 2024 because I always wanted an app that would recommend me local music based on my music taste! #SupportLocal
This repository contains a Python-based web application that leverages the Spotify API to provide personalised local music recommendations. Users can input their Spotify playlists, and the application will suggest Singaporean music based on the characteristics of their chosen playlist. Additionally, users have the option to create a new playlist with these recommendations directly in their Spotify library.

### Features

1. **Playlist Input**: Users can input any of their Spotify playlists to receive recommendations.
2. **Local Music Recommendations**: The algorithm provides song recommendations specifically focused on Singaporean artists and music.
3. **Playlist Creation**: Users can create a new Spotify playlist with the recommended local music directly from this application.
4. **Dynamic Web Interface**: A user-friendly interface built with HTML and Flask for seamless interaction.

### Technology Stack

- **Python**: The core language used for backend processing and interacting with the Spotify API.
- **Flask**: A lightweight WSGI web application framework used to build the web interface.
- **Spotipy**: A lightweight Python library for the Spotify Web API.
- **Pandas**: Used for data manipulation and analysis.
- **Scikit-Learn**: Utilized for feature extraction and cosine similarity calculation.
- **HTML/CSS**: For building and styling the front-end interface.

### How It Works

1. **Authentication**: Users log-in using their Spotify credentials. This is handled securely via the Spotify OAuth.
2. **Playlist Selection**: Users select a playlist from their Spotify library to base the recommendations on.
3. **Feature Extraction**: The application extracts various audio features from the selected playlist using the Spotify API. These features include Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, and Genre.
4. **Recommendation Algorithm**: 
    - The algorithm creates a feature vector for the input playlist.
    - It then compares this vector with the features of local Singaporean tracks using cosine similarity to find the most similar tracks.
    - Recommendations are made based on this similarity score.
5. **Playlist Creation**: Users can choose to create a new Spotify playlist with the recommended tracks, which is then added to their Spotify library.

### Key Components

- **app.py**: The main application file that sets up Flask routes and handles user interactions.
- **templates**: Contains the HTML templates for rendering the web pages.
- **static**: Holds static files like CSS for styling the web interface.
- **spotify_functions.py**: Includes functions for interacting with the Spotify API and processing the playlist data.

### Hosting and Status
- The application is currently hosted on PythonAnywhere.
- An extension request is pending approval by Spotify. The site is not live yet and will go live once Spotify approves the extension request.

### Screenshots of the App
![image](https://github.com/jaylim2216/localify-spotify/assets/98379009/d330d9bc-25ca-4fcc-808d-e49d385c78d7)
![image](https://github.com/jaylim2216/localify-spotify/assets/98379009/7d82673a-6941-42dd-8a2b-d359057490d6)
![image](https://github.com/jaylim2216/localify-spotify/assets/98379009/20fe8b5c-13e3-4865-8d7f-41efeb9ad853)
![image](https://github.com/jaylim2216/localify-spotify/assets/98379009/c83352ff-0c9e-43c4-884f-fadddf8a0010)
![image](https://github.com/jaylim2216/localify-spotify/assets/98379009/ab9d5222-dd4c-43f6-84b2-c3d5b139218c)


### Contributing

I welcome contributions! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to explore the code and enhance it further. Enjoy discovering new local music with Localify!
