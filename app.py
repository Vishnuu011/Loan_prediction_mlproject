from flask import Flask,request,render_template
from loan_prediction.pipline.prediction_pipeline import predictPipline, CustomData



app=Flask(__name__, template_folder="templates")


@app.route("/",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            person_age=float(request.form.get("person_age")),
            person_gender=request.form.get("person_gender"),
            person_education=request.form.get("person_education"),
            person_income=float(request.form.get("person_income")),
            person_emp_exp=int(request.form.get("person_emp_exp")),
            person_home_ownership=request.form.get("person_home_ownership"),
            loan_amnt=float(request.form.get("loan_amnt")),
            loan_intent=request.form.get("loan_intent"),
            loan_int_rate=float(request.form.get("loan_int_rate")),
            loan_percent_income=float(request.form.get("loan_percent_income")),
            cb_person_cred_hist_length=float(request.form.get("cb_person_cred_hist_length")),
            credit_score=int(request.form.get("credit_score")),
            previous_loan_defaults_on_file=request.form.get("previous_loan_defaults_on_file")
        )
        final_data=data.get_data_as_data_frame()

        predict_pipeline=predictPipline()

        pred=predict_pipeline.predict(final_data)



        return render_template("result.html",final_result="rejected" if pred == 0 else "approved")

