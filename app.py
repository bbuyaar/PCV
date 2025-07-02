from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/model.h5')  # Pastikan path ke model benar

# Label kelas sesuai model (30 kelas)
label_map = {
    0: 'Balai Pertolongan Pertama',
    1: 'Banyak Anak Anak',
    2: 'Dilarang Belok Kanan',
    3: 'Dilarang Berhenti',
    4: 'Dilarang Berjalan Terus',
    5: 'Dilarang Masuk',
    6: 'Dilarang Mendahului',
    7: 'Dilarang Parkir',
    8: 'Dilarang Putar Balik',
    9: 'Gereja',
    10: 'Hati Hati',
    11: 'Jalur Penyeberangan',
    12: 'Lampu Lalu Lintas',
    13: 'Larangan Kecepatan - 30 km-jam',
    14: 'Larangan Kecepatan - 40 km-jam',
    15: 'Larangan Kendaraan MST - 10 ton',
    16: 'Masjid',
    17: 'Pemberhentian Bus',
    18: 'Perintah Ikuti Bundaran',
    19: 'Perintah Jalur Sepeda',
    20: 'Perintah Lajur Kiri',
    21: 'Perintah Pilih Satu Jalur',
    22: 'Persimpangan 3 Prioritas',
    23: 'Persimpangan 3 Sisi Kanan Prioritas',
    24: 'Persimpangan 3 sisi Kiri Prioritas',
    25: 'Persimpangan Empat',
    26: 'Putar Balik',
    27: 'Rumah Sakit',
    28: 'SPBU',
    29: 'Tempat Parkir'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join('static/images', file.filename)
            file.save(filepath)

            # Preprocessing gambar
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = model.predict(img_array)
            class_id = int(np.argmax(prediction))
            label = label_map.get(class_id, "Tidak Diketahui")
            confidence = float(np.max(prediction)) * 100  # Tambahkan skor

            return render_template('result.html', label=label, confidence=confidence, image_file=file.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
