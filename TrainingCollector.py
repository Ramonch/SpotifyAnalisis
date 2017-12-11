import spotipy
import spotipy.util as util
import pandas as pd

sp = spotipy.Spotify()

# Claves de acceso a la API de Spotify
client_id = 'id del cliente'
client_secret = 'clave de acceso'

"""Codigo para la obtencion del dataset con canciones postivas"""

# Token de autentificación
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)

playlist = sp.user_playlist('ramonch21', playlist_id='1mFVch763NSXx1fcZV7vQx')
songs = playlist['tracks']['items']

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
assessment = []
for i in range(len(songs)):
    track = songs[i]['track']
    popularity.append(track['popularity'])
    title.append(track['name'])
    album = track['album']
    album_name.append(album['name'])
    artist = track['artists']
    artist_name.append(artist[0]['name'])
    info.append(songs[i]['track']['id'])
    assessment.append(1)

data2 = pd.DataFrame.from_items([('Title', title), ('Artist',artist_name), ('Album', album_name), ('Popularity', popularity)])

data3 = pd.DataFrame.from_items([('Appreciation', assessment)])

# Características técnicas
features = sp.audio_features(info)
data = pd.DataFrame(features)
data.pop('analysis_url')
data.pop('track_href')
data.pop('uri')
data.pop('type')
data.pop('id')

# Crear un dataset final con todos los datos
final_data = data.join(data2).join(data3)

# Transformar los datos en un csv
final_data.to_csv('TrainingPositiveData.csv', columns=None)


"""Codigo para la obtención del dataset con canciones negativas"""


playlist = sp.user_playlist('ramonch21', playlist_id='30Wx8exlQAEC6DQbsf1D9e')
songs = playlist['tracks']['items']

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
assessment = []
for i in range(len(songs)):
    track = songs[i]['track']
    popularity.append(track['popularity'])
    title.append(track['name'])
    album = track['album']
    album_name.append(album['name'])
    artist = track['artists']
    artist_name.append(artist[0]['name'])
    info.append(songs[i]['track']['id'])
    assessment.append(0)

data2 = pd.DataFrame.from_items([('Title', title), ('Artist',artist_name), ('Album', album_name), ('Popularity', popularity)])

data3 = pd.DataFrame.from_items([('Appreciation', assessment)])

# Características técnicas
features = sp.audio_features(info)
data = pd.DataFrame(features)
data.pop('analysis_url')
data.pop('track_href')
data.pop('uri')
data.pop('type')
data.pop('id')

# Crear un dataset final con todos los datos
final_data = data.join(data2).join(data3)

# Transformar los datos en un csv
final_data.to_csv('TrainingNegativeData.csv', columns=None)

"""Codigo para la obtención de dataset con canciones para testear"""

playlist = sp.user_playlist('ramonch21', playlist_id='24A3gUGaJeXsvJnM7o9kKe')
songs = playlist['tracks']['items']

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
assessment = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1]
for i in range(len(songs)):
    track = songs[i]['track']
    popularity.append(track['popularity'])
    title.append(track['name'])
    album = track['album']
    album_name.append(album['name'])
    artist = track['artists']
    artist_name.append(artist[0]['name'])
    info.append(songs[i]['track']['id'])


data2 = pd.DataFrame.from_items([('Title', title), ('Artist',artist_name), ('Album', album_name), ('Popularity', popularity)])

data3 = pd.DataFrame.from_items([('Appreciation', assessment)])

# Características técnicas
features = sp.audio_features(info)
data = pd.DataFrame(features)
data.pop('analysis_url')
data.pop('track_href')
data.pop('uri')
data.pop('type')
data.pop('id')

# Crear un dataset final con todos los datos
final_data = data.join(data2).join(data3)

# Transformar los datos en un csv
final_data.to_csv('TestData.csv', columns=None)