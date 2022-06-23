from flask import send_from_directory
from flask import Flask, flash, request, redirect, url_for, session
from flask import render_template
from url_utils import get_base_url
from werkzeug.utils import secure_filename
import os, shutil
import torch

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)


# if the base url is not empty, then the server is running in development, so specify the static folder to serve static files.
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'abc'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tif', 'tiff'])
OUTPUT_IMAGE = "output.jpg"

# clear upload folder
for filename in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception:
        continue

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 2**25

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt', force_reload=True)

minimum_confidence = 0.25

def check_ext(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route(f'{base_url}', methods=['GET', 'POST'])
def sendToHome():
    return redirect(url_for('home'))
@app.route(f'{base_url}/home', methods=['GET', 'POST'])
def home():
    return render_template('mainhome.html')
@app.route(f'{base_url}/timeline', methods=['GET', 'POST'])
def timeline():
    return render_template('timeline.html')
@app.route(f'{base_url}/teampage', methods=['GET', 'POST'])
def teampage():
    return render_template('teampage.html')

@app.route(f'{base_url}/detect', methods=['GET', 'POST'])
def detect():
    global minimum_confidence
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # check extension
        if file and check_ext(file.filename):
            minimum_confidence = int(request.form['confidence']) / 100
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('detect.html')

@app.route(f'{base_url}/uploads/upload', methods=['GET', 'POST'])
def upload():
    global minimum_confidence
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # check extension
        if file and check_ext(file.filename):
#             minimum_confidence = 0.25
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('detect.html')


# when a file appears in static/uploads/
@app.route(f'{base_url}/uploads/<filename>')
def uploaded_file(filename):
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)
    model.conf = minimum_confidence
    results = model(image_path, size=416)
    if len(results.pandas().xyxy) > 0:
        # results.print()
        save_dir = os.path.join(here, app.config['UPLOAD_FOLDER'])
        results.save(save_dir=save_dir)
        confidences = list(results.pandas().xyxy[0]['confidence'])
        format_confidences = []
        # convert to nearest percent as string
        for decimal in confidences:
            format_confidences.append(str(round(decimal * 100)) + '%')
        # format the labels to sort, capitalize
        labels = set(list(results.pandas().xyxy[0]['name']))
        labels = [car.capitalize() for car in labels]
        # join items with 'and' between them
        def and_syntax(str_list):
            if len(str_list) == 1:
                return "".join(str_list)
            if len(str_list) == 2:
                return " and ".join(str_list)
            if len(str_list) > 2:
                str_list[-1] = "and " + str_list[-1]
                return ", ".join(str_list)
            return ''
        format_confidences = and_syntax(format_confidences)
        labels = and_syntax(labels)
        # model output filename should always have .jpg
        new_filename, extension = os.path.splitext(image_path)
        new_filename += ".jpg"
        output_path = os.path.join(here, app.config['UPLOAD_FOLDER'], OUTPUT_IMAGE)
        os.rename(new_filename, output_path)
        if len(confidences) > 0:
            return render_template('results.html',
                                   confidences=format_confidences,
                                   labels=labels,
                                   filename=OUTPUT_IMAGE,
                                   min_conf=(minimum_confidence * 100))
        else:
            return render_template('results.html',
                                   labels='no cars',
                                   confidences='100%',
                                   filename=OUTPUT_IMAGE,
                                   min_conf=minimum_confidence*100)
    found = False
    return render_template('results.html',
                           labels='no cars',
                           confidences='high',
                           filename=filename,
                           min_conf=minimum_confidence * 100)

@app.route(f'{base_url}/uploads/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc10.ai-camp.dev'
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
