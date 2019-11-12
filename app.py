from flask import Flask, render_template, jsonify, request
from main.main import printarray
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 7, type=int)
    b = request.args.get('b', 7, type=int)
    return jsonify(result=a + b)

@app.route("/about")
def about():
    return render_template("about.html")
@app.route('/data', methods=['POST'])
def get_names():
    if request.method == 'POST':
        names = request.get_json()
        # print(names['data'].values())
        data = printarray(list(names['data'].values()))
    return data
@app.route('/user/<name>')
def show_user_profile(name):
    return render_template('tun.html', name = name)
@app.route('/hello/<user>')
def hello_name(user):
   return render_template('hello.html', name = user)
if __name__ == "__main__":
    app.run(debug=True)