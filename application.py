from flask import Flask, request, render_template
from src.component.user_data_prediction import ProcessUserData, Prediction

application = Flask(__name__)
app = application

# add route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])    
def predict_user_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        user_data = ProcessUserData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score =  float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        df_user_input = user_data.get_data_frame()
        prediction_obj = Prediction()
        result = prediction_obj.get_prediction(df_user_input)
        return render_template('home.html', results=round(result[0], 2))
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000) 