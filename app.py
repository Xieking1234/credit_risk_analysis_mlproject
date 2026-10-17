from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            loan_amnt=float(request.form.get('loan_amnt')),
            funded_amnt=float(request.form.get('funded_amnt')),
            funded_amnt_inv=float(request.form.get('funded_amnt_inv')),
            term=request.form.get('term'),
            int_rate=float(request.form.get('int_rate')),
            installment=float(request.form.get('installment')),
            sub_grade=request.form.get('sub_grade'),
            annual_inc=float(request.form.get('annual_inc')),
            verification_status=request.form.get('verification_status'),
            dti=float(request.form.get('dti')),
            delinq_2yrs=float(request.form.get('delinq_2yrs')),
            inq_last_6mths=float(request.form.get('inq_last_6mths')),
            open_acc=float(request.form.get('open_acc')),
            pub_rec=float(request.form.get('pub_rec')),
            revol_bal=float(request.form.get('revol_bal')),
            revol_util=float(request.form.get('revol_util')),
            total_acc=float(request.form.get('total_acc')),
            initial_list_status=request.form.get('initial_list_status'),
            out_prncp=float(request.form.get('out_prncp')),
            out_prncp_inv=float(request.form.get('out_prncp_inv')),
            total_pymnt=float(request.form.get('total_pymnt')),
            total_pymnt_inv=float(request.form.get('total_pymnt_inv')),
            total_rec_prncp=float(request.form.get('total_rec_prncp')),
            total_rec_int=float(request.form.get('total_rec_int')),
            total_rec_late_fee=float(request.form.get('total_rec_late_fee')),
            recoveries=float(request.form.get('recoveries')),
            collection_recovery_fee=float(request.form.get('collection_recovery_fee')),
            last_pymnt_amnt=float(request.form.get('last_pymnt_amnt')),
            collections_12_mths_ex_med=float(request.form.get('collections_12_mths_ex_med')),
            tot_coll_amt=float(request.form.get('tot_coll_amt')),
            tot_cur_bal=float(request.form.get('tot_cur_bal')),
            total_rev_hi_lim=float(request.form.get('total_rev_hi_lim')),
            home_ownership=request.form.get('home_ownership'),
            purpose=request.form.get('purpose')
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")