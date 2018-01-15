import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ignorar los warnings de Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Leer los datos de estrenamiento
positive_data = pd.read_csv('TrainingPositiveData.csv')
negative_data = pd.read_csv('TrainingNegativeData.csv')
training_data = pd.concat([positive_data, negative_data])

training_data.pop("Title")
training_data.pop("Artist")
training_data.pop("Album")
training_data.pop("duration_ms")

X_training = training_data.drop('Appreciation', axis = 1).values
Y_training = training_data[['Appreciation']].values

test_data = pd.read_csv('TestData.csv')

test_data.pop("Title")
test_data.pop("Artist")
test_data.pop("Album")
test_data.pop("duration_ms")

X_test = test_data.drop('Appreciation', axis = 1).values
Y_test = test_data[['Appreciation']].values

# Escalar los datos para que la red trabaje bien
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_training_escalado = X_scaler.fit_transform(X_training)
Y_training_escalado = Y_scaler.fit_transform(Y_training)

X_test_escalado = Y_scaler.fit_transform(X_test)
Y_test_escalado = Y_scaler.fit_transform(Y_test)

#Parametros del modelo
inputs = 14
outputs = 1

learning_rate = 0.01
bucle_entrenamiento = 40
n_neurons1 = 1500
n_neurons2 = 800
n_neurons3 = 500
n_neurons4 = 200
n_neurons5 = 80
n_neurons6 = 40
n_neurons7 = 15
n_neurons8 = 8

""" Definir las capas de la red neuronal """

#   Capa de entrada
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, inputs))

# Primera capa
with tf.variable_scope('Capa1'):
    weights = tf.get_variable("weights1", shape=[inputs, n_neurons1], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[n_neurons1], initializer=tf.zeros_initializer())
    capa1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Segunda capa
with tf.variable_scope('Capa2'):
    weights = tf.get_variable("weights2", shape=[n_neurons1, n_neurons2], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[n_neurons2], initializer=tf.zeros_initializer())
    capa2_output = tf.nn.relu(tf.matmul(capa1_output, weights) + biases)

# Tercera capa
with tf.variable_scope('Capa3'):
    weights = tf.get_variable("weights3", shape=[n_neurons2, n_neurons3], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[n_neurons3], initializer=tf.zeros_initializer())
    capa3_output = tf.nn.relu(tf.matmul(capa2_output, weights) + biases)

# Cuarta capa
with tf.variable_scope('Capa4'):
    weights = tf.get_variable("weights4", shape=[n_neurons3, n_neurons4], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[n_neurons4], initializer=tf.zeros_initializer())
    capa4_output = tf.nn.relu(tf.matmul(capa3_output, weights) + biases)

# Quinta capa
with tf.variable_scope('Capa5'):
    weights = tf.get_variable("weights5", shape=[n_neurons4, n_neurons5], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases5", shape=[n_neurons5], initializer=tf.zeros_initializer())
    capa5_output = tf.nn.relu(tf.matmul(capa4_output, weights) + biases)

# Sexta capa
with tf.variable_scope('Capa6'):
    weights = tf.get_variable("weights6", shape=[n_neurons5, n_neurons6], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases6", shape=[n_neurons6], initializer=tf.zeros_initializer())
    capa6_output = tf.nn.relu(tf.matmul(capa5_output, weights) + biases)

# Septima capa
with tf.variable_scope('Capa7'):
    weights = tf.get_variable("weights7", shape=[n_neurons6, n_neurons7], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases7", shape=[n_neurons7], initializer=tf.zeros_initializer())
    capa7_output = tf.nn.relu(tf.matmul(capa6_output, weights) + biases)
# Octaba capa
with tf.variable_scope('Capa8'):
    weights = tf.get_variable("weights8", shape=[n_neurons7, n_neurons8], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases8", shape=[n_neurons8], initializer=tf.zeros_initializer())
    capa8_output = tf.nn.relu(tf.matmul(capa7_output, weights) + biases)
# Cpara final
with tf.variable_scope('Capa9'):
    weights = tf.get_variable("weights9", shape=[n_neurons8, outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases9", shape=[outputs], initializer=tf.zeros_initializer())
    prediccion = tf.nn.relu(tf.matmul(capa8_output, weights) + biases)

# Definir la funcion de coste
with tf.variable_scope('Coste'):
    Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
    coste = tf.reduce_mean(tf.squared_difference(prediccion, Y))

# Funcion de optimización
with tf.variable_scope('Entrenamiento'):
    optimizador = tf.train.AdamOptimizer(learning_rate).minimize(coste)

# Loggin de para debuguear
with tf.variable_scope('logging'):
    tf.summary.scalar('CosteActual', coste)
    summary = tf.summary.merge_all()

# Iniciar la sesion para las operaciones de Tensorflow
with tf.Session() as sesion:
    sesion.run(tf.global_variables_initializer())

    # Crear archivos de log para guardar el proceso de entrenamiento
    training_log = tf.summary.FileWriter("./logs/training", sesion.graph)
    testing_log = tf.summary.FileWriter("./logs/testing", sesion.graph)

    # Ejecutar un bucle para que el optimizador entrene la red
    for paso in range (bucle_entrenamiento):

        # Darle los datos
        sesion.run(optimizador, feed_dict={X: X_training_escalado, Y: Y_training_escalado})

        # Cada cinco pasos del bucle imprimir el estado del entrenamiento
        if paso % 2 == 0:

            training_cost, training_summary = sesion.run([coste, summary], feed_dict={X: X_training_escalado, Y: Y_training_escalado})
            testing_cost, testing_summary = sesion.run([coste, summary], feed_dict={X: X_test_escalado, Y: Y_test_escalado})

            # Escribir el estado actual en el log correspondiente
            training_log.add_summary(training_summary, paso)
            testing_log.add_summary(testing_summary, paso)

            #Imprimir el estado actual
            print("Paso: {} -- Coste entrenamiento: {} -- Coste test: {}".format(paso, training_cost, testing_cost))

    # Obtener los valores del coste operacional del entrenamiento y el testeo
    final_training_cost = sesion.run(coste, feed_dict={X: X_training_escalado, Y: Y_training_escalado})
    final_testing_cost = sesion.run(coste, feed_dict={X: X_test_escalado, Y: Y_test_escalado})

    print("Coste de entrenamiento final: {}".format(final_training_cost))
    print("Coste de test finalt: {}".format(final_testing_cost))

    # PRediccion con los datos de testeo
    Y_prediccion_escalado = sesion.run(prediccion, feed_dict={X: X_test_escalado})
    #Y_prediccion = Y_scaler.inverse_transform(Y_prediccion_escalado)

    for i in range (len(test_data)):
        valoracion_real = test_data['Appreciation'].values[i]
        valoracion_predecida = Y_prediccion_escalado[i][0]

        valoracion_predecida = float("{0:.2f}".format(valoracion_predecida))
        if valoracion_predecida>1:
            valoracion_predecida = 1

        print("Valoracion real: {} --- Valoracion predecida: {}".format(valoracion_real, valoracion_predecida))


    # Aplicar el modelo a los datos
    playlist = pd.read_csv('Playlist.csv')

    title = playlist.pop("Title")
    title = title.to_frame(name=None)
    artist = playlist.pop("Artist")
    artist = artist.to_frame(name=None)
    playlist.pop("Album")
    playlist.pop("duration_ms")

    playlist_escalado = X_scaler.fit_transform(playlist)

    valoracion = sesion.run(prediccion, feed_dict={X: playlist_escalado})
    val = []
    for i in range (len(valoracion)):
        valoracion_predecida = valoracion[i][0]
        valoracion_predecida = float("{0:.2f}".format(valoracion_predecida))
        if valoracion_predecida > 1:
            valoracion_predecida = 1
        val.append(valoracion_predecida)

    appreciation = pd.DataFrame.from_items([('Appreciation', val)])

    data = title.join(artist).join(appreciation)
    data.to_csv('ValoresPredecidos.csv', columns=None)
