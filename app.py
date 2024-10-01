from flask import Flask, request, render_template
import os
from itworks import process_cv
import logging

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    # Log request info
    app.logger.debug('Received request on /upload')

    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return render_template("index.html", insights=None), 400

    file = request.files['file']
    app.logger.debug(f'File received: {file.filename}')

    if file.filename == '':
        app.logger.error('No file selected')
        return render_template("index.html", insights=None), 400

    if not allowed_file(file.filename):
        app.logger.error(f'Invalid file extension for file: {file.filename}')
        return render_template("index.html", insights=None), 400

    # Save the file temporarily to the temp directory
    temp_dir = os.getenv('TEMP') or os.path.join(os.environ['USERPROFILE'], 'AppData', 'Local', 'Temp')
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    app.logger.debug(f'File saved to {file_path}')

    try:
        # Call the process_cv function from itworks.py to process the file
        insights = process_cv(file_path)
        app.logger.debug(f'Insights processed: {insights}')

        # Render the insights using a proper HTML template
        return render_template('index.html', insights=insights)

    except Exception as e:
        app.logger.error(f'Error processing file: {str(e)}')
        return render_template('index.html', insights=None), 500
    
    finally:
        # Clean up the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            app.logger.debug(f'File removed: {file_path}')

if __name__ == '__main__':
    app.run(debug=True)