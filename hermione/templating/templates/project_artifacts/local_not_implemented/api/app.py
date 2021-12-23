from flask import Flask, request, redirect, url_for, flash, jsonify
import logging

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    resp = jsonify(success=True)
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0')