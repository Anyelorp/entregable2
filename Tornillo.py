from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import pandas as pd

app = Flask(__name__)

# Datos de ejemplo para el árbol de decisiones
data = {
    'Edad': [18, 19, 20, 21, 22, 23, 24, 25, 30, 22, 35, 40, 28, 32, 45],
    'Compras_Mensuales': [2, 3, 1, 4, 5, 2, 3, 6, 4, 5, 2, 3, 6, 1, 4],
    'Producto': ['clavos', 'martillo', 'goma', 'clavos', 'martillo', 'goma', 'clavos', 'martillo', 'goma', 'clavos', 'martillo', 'goma', 'clavos', 'martillo', 'goma'],
    'Direccion': ['Calle A', 'Calle B', 'Calle C', 'Calle D', 'Calle E', 'Calle F', 'Calle G', 'Calle H', 'Calle I', 'Calle J', 'Calle K', 'Calle L', 'Calle M', 'Calle N', 'Calle O'],
    'ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    'Compro_Recientemente': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]  
}

df = pd.DataFrame(data)

# Separar datos en características (X) y etiquetas (y)
X = df[['Edad', 'Compras_Mensuales', 'ID']]
y = df['Compro_Recientemente']

# Crear el modelo de árbol de decisiones con entropía como criterio
model = DecisionTreeClassifier(criterion='entropy')

# Entrenar el modelo
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    edad = int(request.form['edad'])
    compras_mensuales = int(request.form['compras_mensuales'])
    id_cliente = int(request.form['id_cliente'])

    # Realizar la predicción utilizando el modelo de árbol de decisiones
    resultado_prediccion = model.predict([[edad, compras_mensuales, id_cliente]])

    return render_template('resultado.html', resultado=resultado_prediccion[0])

if __name__ == '__main__':
    app.run(debug=True)
