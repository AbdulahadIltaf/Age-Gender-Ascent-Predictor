@app.route('/record', methods=['POST'])
def record_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'}), 400
    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({'error': 'No selected audio'}), 400

    
    # Audio saved successfully
    return jsonify({'message': f'Audio recorded successfully: {filepath}'})