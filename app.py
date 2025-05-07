from flask import Flask, request, render_template, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
UPLOAD_FOLDER = './document/'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect(url_for('upload_file'))
        else:
            flash('Allowed file type is pdf')
            return redirect(request.url)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True) 