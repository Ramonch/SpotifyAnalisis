import spotipy
import spotipy.util as util
import pandas as pd

sp = spotipy.Spotify()

# Claves de acceso a la API de Spotify
client_id = '1b429f00304d4436bfd92b62407c968c'
client_secret = '5f4e95fced7a4c60b38c7af9082cbd23'

"""Codigo para la obtencion del dataset con canciones postivas"""

# Token de autentificación
token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
assessment = []
canciones = 100
contador = 0
bucle = 0

id_usuario = 'ramonch21'
id_positivas = '1mFVch763NSXx1fcZV7vQx'

# Bucle para recolectar la totalidad de canciones de una playlist
data = pd.DataFrame()
while (canciones>=100):
    dataAux = pd.DataFrame()
    bucle = bucle + 1
    canciones = 0
    info = []

    # Indicar de que playlist obtener los datos
    # playlist = sp.user_playlist('118531625', playlist_id='21xZfbKhrSLKbJO4Aa4jG6')
    playlist = sp.user_playlist_tracks(id_usuario, playlist_id=id_positivas, offset=contador)
    # songs = playlist['tracks']['items']
    songs = playlist['items']
    if (len(songs) == 0): break

    for i in range(len(songs)):
        canciones = canciones + 1
        track = songs[i]['track']
        popularity.append(track['popularity'])
        title.append(track['name'])
        album = track['album']
        album_name.append(album['name'])
        artist = track['artists']
        artist_name.append(artist[0]['name'])
        if (songs[i]['track']['id']==None):
            info.append(songs[i-1]['track']['id'])
        else:
            info.append(songs[i]['track']['id'])
        contador = contador + 1
        assessment.append(1)

    features = sp.audio_features(info)
    dataAux = pd.DataFrame(features)
    data = pd.concat([data, dataAux], ignore_index='true')

data2 = pd.DataFrame.from_items([('Title', title), ('Artist',artist_name), ('Album', album_name), ('Popularity', popularity)])

data3 = pd.DataFrame.from_items([('Appreciation', assessment)])

# Características técnicas
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

id_negativas = '30Wx8exlQAEC6DQbsf1D9e'

playlist = sp.user_playlist(id_usuario, playlist_id=id_negativas)
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

id_test = '24A3gUGaJeXsvJnM7o9kKe'

playlist = sp.user_playlist(id_usuario, playlist_id=id_test)
songs = playlist['tracks']['items']

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
assessment = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
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