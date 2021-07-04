from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
app = Flask(__name__)

dic = {0 : 'Gasoline Can', 1 : 'Hammer', 2 : 'Pliers', 3: 'Rope', 4: 'Screw Driver', 5: 'Tool Box', 6: 'Wrench'}

model = load_model('Model/')

model.make_predict_function()

def predict_label(img_path):
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64,64))
	imgArray = tf.keras.preprocessing.image.img_to_array(img)
	compatibleArray = np.expand_dims(imgArray, axis=0)
	p = model.predict_classes(compatibleArray)
	return dic[p[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
