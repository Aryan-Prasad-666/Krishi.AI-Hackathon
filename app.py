from flask import Flask, render_template, request, jsonify
import json
import os
import re
import requests
import base64
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)

gemini_api_key = os.getenv('GEMINI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')
mem0_api_key = os.getenv('MEM0_API_KEY')



@app.route('/')
def index():
    return render_template('home.html')



@app.route('/telegram')
def telegram():
    return render_template('telegram.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)