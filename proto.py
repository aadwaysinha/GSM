# Import packages
from flask import Flask, jsonify, request, redirect, render_template
import numpy as np
import pandas as pd
import math
import sys
import os
import json
from tqdm import tqdm

# Init app
app = Flask(__name__)
app.secret_key = 'monkeyclimbsatree,dontmesswithme'


#Components
@app.route('/')
def ChooseUser():
    return render_template('ChooseUser.html')


# Flask setup
base_dir = os.path.abspath(os.path.dirname(__file__))

# Allowed file types
# Change to any
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}



# Cache control
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload file
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Check if a valid file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Valid file uploaded
            return dummy_method(file)

    # Valid file not uploaded
    return render_template('index.html')

# Dummy method
def dummy_method(file_stream):

    # Perform op
    print("In dummy..")

    # return jsonify("In dummy..")
    return render_template('index.html')

# Main method
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
