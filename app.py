from flask import Flask, render_template_string, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('IRISMODEL.pkl', 'rb'))

index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Flower Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color:white;
        }

        header {
            background-color: #90EE90;
            color: red;
            padding: 10px;
            text-align: center;
        }

        main {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        footer {
            text-align: center;
            padding: 10px;
            background-color: #90EE90;
            color: red;
            position: fixed;
            bottom: 0;
            width: 100%;
        }

        form {
            margin-top: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
        }

        button {
            background-color: #333;
            color: #fff;
            padding: 10px;
            border: none;
            cursor: pointer;
        }

        #predictedResult {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <h1>Iris Flower Prediction</h1>
    </header>
    <main>
        <p>Welcome to the Iris Flower Prediction Model!</p>
        <form id="predictionForm" method="post" action="/predict">
            <label for="sepal_length">Sepal Length:</label>
            <input type="text" id="sepal_length" name="sepal_length" required>

            <label for="sepal_width">Sepal Width:</label>
            <input type="text" id="sepal_width" name="sepal_width" required>

            <label for="petal_length">Petal Length:</label>
            <input type="text" id="petal_length" name="petal_length" required>

            <label for="petal_width">Petal Width:</label>
            <input type="text" id="petal_width" name="petal_width" required>

            <button type="submit">Predict</button>
        </form>
        <div id="predictedResult">{{ result }}</div>
    </main>
    <footer>
        <p>&copy; 2024 Iris Flower Prediction. All rights reserved.</p>
    </footer>
</body>
</html>
"""

@app.route('/')
def home():
    result = ""
    return render_template_string(index_html, result=result)

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    return render_template_string(index_html, result=result)

if __name__ == '__main__':
    app.run(debug=True)
