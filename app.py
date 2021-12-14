from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np


app = Flask(__name__)

# use post and get route


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        # open model file
        with open('knn_pickle', 'rb') as r:
            model = pickle.load(r)

        # input all data in form
        asal = float(request.form['asal'])
        tujuan = float(request.form['tujuan'])
        kabandara = float(request.form['kabandara'])
        bulan = float(request.form['bulan'])
        tanggal = float(request.form['tanggal'])
        tahun = float(request.form['tahun'])

        # change data into array
        datas = np.array((asal, tujuan, kabandara, bulan, tanggal, tahun))

        # change shape array into 2Dimention
        datas = np.reshape(datas, (1, -1))

        finalResult = model.predict(datas)

        return render_template('hasil.html', finalData=finalResult)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
