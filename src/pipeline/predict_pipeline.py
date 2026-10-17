import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

def decode_loan_status(pred_value):
    status_mapping = {
        0: 'Charged Off',
        1: 'Current',
        2: 'Default',
        3: 'Does not meet the credit policy. Status:Charged Off',
        4: 'Does not meet the credit policy. Status:Fully Paid',
        5: 'Fully Paid',
        6: 'In Grace Period',
        7: 'Issued',
        8: 'Late (16-30 days)',
        9: 'Late (31-120 days)'
    }
    if hasattr(pred_value, '__iter__'):
        return [status_mapping.get(int(val), 'Unknown') for val in pred_value]
    return status_mapping.get(int(pred_value), 'Unknown')


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            scaler = preprocessor["scaler"]

            # Step 1: Scale features using MinMaxScaler
            data_scaled = scaler.transform(features)

            # Step 2: Make predictions with XGBoost
            preds = model.predict(data_scaled)
            
            # Step 3: Decode numeric prediction to text label
            return decode_loan_status(preds)
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        loan_amnt: float,
        funded_amnt: float,
        funded_amnt_inv: float,
        term: str,
        int_rate: float,
        installment: float,
        sub_grade: str,
        annual_inc: float,
        verification_status: str,
        dti: float,
        delinq_2yrs: float,
        inq_last_6mths: float,
        open_acc: float,
        pub_rec: float,
        revol_bal: float,
        revol_util: float,
        total_acc: float,
        initial_list_status: str,
        out_prncp: float,
        out_prncp_inv: float,
        total_pymnt: float,
        total_pymnt_inv: float,
        total_rec_prncp: float,
        total_rec_int: float,
        total_rec_late_fee: float,
        recoveries: float,
        collection_recovery_fee: float,
        last_pymnt_amnt: float,
        collections_12_mths_ex_med: float,
        tot_coll_amt: float,
        tot_cur_bal: float,
        total_rev_hi_lim: float,
        home_ownership: str,
        purpose: str
    ):
        self.loan_amnt = loan_amnt
        self.funded_amnt = funded_amnt
        self.funded_amnt_inv = funded_amnt_inv
        self.term = term
        self.int_rate = int_rate
        self.installment = installment
        self.sub_grade = sub_grade
        self.annual_inc = annual_inc
        self.verification_status = verification_status
        self.dti = dti
        self.delinq_2yrs = delinq_2yrs
        self.inq_last_6mths = inq_last_6mths
        self.open_acc = open_acc
        self.pub_rec = pub_rec
        self.revol_bal = revol_bal
        self.revol_util = revol_util
        self.total_acc = total_acc
        self.initial_list_status = initial_list_status
        self.out_prncp = out_prncp
        self.out_prncp_inv = out_prncp_inv
        self.total_pymnt = total_pymnt
        self.total_pymnt_inv = total_pymnt_inv
        self.total_rec_prncp = total_rec_prncp
        self.total_rec_int = total_rec_int
        self.total_rec_late_fee = total_rec_late_fee
        self.recoveries = recoveries
        self.collection_recovery_fee = collection_recovery_fee
        self.last_pymnt_amnt = last_pymnt_amnt
        self.collections_12_mths_ex_med = collections_12_mths_ex_med
        self.tot_coll_amt = tot_coll_amt
        self.tot_cur_bal = tot_cur_bal
        self.total_rev_hi_lim = total_rev_hi_lim
        self.home_ownership = home_ownership
        self.purpose = purpose

    def _sub_grades_encoding(self, x: str) -> float:
        x = str(x).upper().strip()
        val = 0.0
        if 'A' in x: val = 7.0
        elif 'B' in x: val = 6.0
        elif 'C' in x: val = 5.0
        elif 'D' in x: val = 4.0
        elif 'E' in x: val = 3.0
        elif 'F' in x: val = 2.0
        elif 'G' in x: val = 1.0

        if '1' in x: val += 0.8
        elif '2' in x: val += 0.6
        elif '3' in x: val += 0.4
        elif '4' in x: val += 0.2
        elif '5' in x: val += 0.0
        return val

    def get_data_as_data_frame(self):
        try:
            # 1. Encode term (' 36 months' -> 0.0, ' 60 months' -> 1.0)
            term_encoded = 0.0 if "36" in str(self.term) else 1.0

            # 2. Encode verification_status ('Not Verified' -> 0.0, else 1.0)
            verification_encoded = 0.0 if str(self.verification_status).strip() == 'Not Verified' else 1.0

            # 3. Encode initial_list_status ('f' -> 0.0, 'w' -> 1.0)
            initial_list_encoded = 0.0 if str(self.initial_list_status).strip().lower() == 'f' else 1.0

            # 4. Encode sub_grade ('B2' -> 6.6)
            sub_grade_encoded = self._sub_grades_encoding(self.sub_grade)

            # Build base numeric features
            custom_data_input_dict = {
                "loan_amnt": [float(self.loan_amnt)],
                "funded_amnt": [float(self.funded_amnt)],
                "funded_amnt_inv": [float(self.funded_amnt_inv)],
                "term": [float(term_encoded)],
                "int_rate": [float(self.int_rate)],
                "installment": [float(self.installment)],
                "sub_grade": [float(sub_grade_encoded)],
                "annual_inc": [float(self.annual_inc)],
                "verification_status": [float(verification_encoded)],
                "dti": [float(self.dti)],
                "delinq_2yrs": [float(self.delinq_2yrs)],
                "inq_last_6mths": [float(self.inq_last_6mths)],
                "open_acc": [float(self.open_acc)],
                "pub_rec": [float(self.pub_rec)],
                "revol_bal": [float(self.revol_bal)],
                "revol_util": [float(self.revol_util)],
                "total_acc": [float(self.total_acc)],
                "initial_list_status": [float(initial_list_encoded)],
                "out_prncp": [float(self.out_prncp)],
                "out_prncp_inv": [float(self.out_prncp_inv)],
                "total_pymnt": [float(self.total_pymnt)],
                "total_pymnt_inv": [float(self.total_pymnt_inv)],
                "total_rec_prncp": [float(self.total_rec_prncp)],
                "total_rec_int": [float(self.total_rec_int)],
                "total_rec_late_fee": [float(self.total_rec_late_fee)],
                "recoveries": [float(self.recoveries)],
                "collection_recovery_fee": [float(self.collection_recovery_fee)],
                "last_pymnt_amnt": [float(self.last_pymnt_amnt)],
                "collections_12_mths_ex_med": [float(self.collections_12_mths_ex_med)],
                "tot_coll_amt": [float(self.tot_coll_amt)],
                "tot_cur_bal": [float(self.tot_cur_bal)],
                "total_rev_hi_lim": [float(self.total_rev_hi_lim)],
            }

            df = pd.DataFrame(custom_data_input_dict)

            # 5. One-Hot Encode home_ownership columns
            home_ownership_cols = [
                'home_ownership_ANY', 'home_ownership_MORTGAGE', 'home_ownership_NONE',
                'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT'
            ]
            selected_home = str(self.home_ownership).strip().upper()
            for col in home_ownership_cols:
                df[col] = 1.0 if col.endswith(selected_home) else 0.0

            # 6. One-Hot Encode purpose columns
            purpose_cols = [
                'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
                'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical',
                'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                'purpose_vacation', 'purpose_wedding'
            ]
            selected_purpose = str(self.purpose).strip().lower()
            for col in purpose_cols:
                df[col] = 1.0 if col == f"purpose_{selected_purpose}" else 0.0

            return df

        except Exception as e:
            raise CustomException(e, sys)