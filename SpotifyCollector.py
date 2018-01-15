import spotipy
import spotipy.util as util
import pandas as pd

sp = spotipy.Spotify()

# Claves de acceso a la API de Spotify
client_id = '1b429f00304d4436bfd92b62407c968c'
client_secret = '5f4e95fced7a4c60b38c7af9082cbd23'

token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)

# Obtener las caracteristicas de las canciones
info = []
title = []
artist_name = []
popularity = []
album_name = []
canciones = 100
contador = 0
bucle = 0

# Bucle para recolectar la totalidad de canciones de una playlist
data = pd.DataFrame()
id_usuario = '118531625'
id_playlist = '21xZfbKhrSLKbJO4Aa4jG6'
while (canciones>=100):
    dataAux = pd.DataFrame()
    bucle = bucle + 1
    canciones = 0
    info = []

    # Indicar de que playlist obtener los datos
    #playlist = sp.user_playlist('118531625', playlist_id='21xZfbKhrSLKbJO4Aa4jG6')
    playlist = sp.user_playlist_tracks(id_usuario, playlist_id=id_playlist, offset=contador)
    #songs = playlist['tracks']['items']
    songs = playlist['items']
    if (len(songs)==0): break

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

    features = sp.audio_features(info)
    dataAux = pd.DataFrame(features)
    data = pd.concat([data, dataAux], ignore_index='true')

data2 = pd.DataFrame.from_items([('Title', title), ('Artist',artist_name), ('Album', album_name), ('Popularity', popularity)])

# Características técnicas
#features = sp.audio_features(info)
#data = pd.DataFrame(features)
data.pop('analysis_url')
data.pop('track_href')
data.pop('uri')
data.pop('type')
data.pop('id')

# Crear un dataset final con todos los datos
final_data = data2.join(data)

# Transformar los datos en un csv
final_data.to_csv('Playlist.csv', columns=None)