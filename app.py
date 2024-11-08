from flask import Flask, render_template, request, redirect, url_for, flash, abort, current_app, send_from_directory


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/artikel-penyakit')
def artikel():
    return render_template('pages/artikel-penyakit.html')

@app.route('/rekomendasi-pengobatan')
def rekomendasi():
    return render_template('pages/rekomendasi-pengobatan.html')
@app.route('/wawasan-dan-tips')
def wawasan():
    return render_template('pages/wawasan-dan-tips.html')  
    
    
if __name__ == '__main__':
    app.run(debug=True)