from flask import Flask, request, render_template_string
from movie_recommender 

app = Flask(__name__)

# HTML template for the input form and output display
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Input Processor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { text-align: center; }
        input[type="text"] { padding: 8px; width: 200px; }
        input[type="submit"] { padding: 8px 16px; margin-left: 10px; }
        .output { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Input Processor</h1>
        <form method="POST" action="/">
            <input type="text" name="user_input" placeholder="Enter your input" required>
            <input type="submit" value="Submit">
        </form>
        {% if output %}
            <div class="output">
                <h3>Output:</h3>
                <p>{{ output }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        try:
            output = recommend_movies(user_input,user_movie_matrix,5)
            
        except Exception as e:
            output = user_input
    return render_template_string(HTML_TEMPLATE, output=output)

if __name__ == '__main__':
    app.run(debug=True)