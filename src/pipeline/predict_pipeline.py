import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Extract scaler and pca from the saved dictionary
            scaler = preprocessor["scaler"]
            pca = preprocessor["pca"]

            # Step 1: Scale features using MinMaxScaler
            data_scaled = scaler.transform(features)
            
            # Step 2: Reduce dimensions using PCA (25 components)
            data_pca = pca.transform(data_scaled)

            # Step 3: Make predictions with XGBoost
            preds = model.predict(data_pca)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        loan_amnt: float,
        funded_amnt: float,
        funded_amnt_inv: float,
        term: float,
        int_rate: float,
        installment: float,
        sub_grade: float,
        annual_inc: float,
        verification_status: float,
        dti: float,
        delinq_2yrs: float,
        inq_last_6mths: float,
        open_acc: float,
        pub_rec: float,
        revol_bal: float,
        revol_util: float,
        total_acc: float,
        initial_list_status: float,
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
        home_ownership_ANY: float,
        home_ownership_MORTGAGE: float,
        home_ownership_NONE: float,
        home_ownership_OTHER: float,
        home_ownership_OWN: float,
        home_ownership_RENT: float,
        purpose_car: float,
        purpose_credit_card: float,
        purpose_debt_consolidation: float,
        purpose_educational: float,
        purpose_home_improvement: float,
        purpose_house: float,
        purpose_major_purchase: float,
        purpose_medical: float,
        purpose_moving: float,
        purpose_other: float,
        purpose_renewable_energy: float,
        purpose_small_business: float,
        purpose_vacation: float,
        purpose_wedding: float
    ):
        # Assigning all features from your dataset columns (excluding loan_status)
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
        self.home_ownership_ANY = home_ownership_ANY
        self.home_ownership_MORTGAGE = home_ownership_MORTGAGE
        self.home_ownership_NONE = home_ownership_NONE
        self.home_ownership_OTHER = home_ownership_OTHER
        self.home_ownership_OWN = home_ownership_OWN
        self.home_ownership_RENT = home_ownership_RENT
        self.purpose_car = purpose_car
        self.purpose_credit_card = purpose_credit_card
        self.purpose_debt_consolidation = purpose_debt_consolidation
        self.purpose_educational = purpose_educational
        self.purpose_home_improvement = purpose_home_improvement
        self.purpose_house = purpose_house
        self.purpose_major_purchase = purpose_major_purchase
        self.purpose_medical = purpose_medical
        self.purpose_moving = purpose_moving
        self.purpose_other = purpose_other
        self.purpose_renewable_energy = purpose_renewable_energy
        self.purpose_small_business = purpose_small_business
        self.purpose_vacation = purpose_vacation
        self.purpose_wedding = purpose_wedding

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "loan_amnt": [self.loan_amnt],
                "funded_amnt": [self.funded_amnt],
                "funded_amnt_inv": [self.funded_amnt_inv],
                "term": [self.term],
                "int_rate": [self.int_rate],
                "installment": [self.installment],
                "sub_grade": [self.sub_grade],
                "annual_inc": [self.annual_inc],
                "verification_status": [self.verification_status],
                "dti": [self.dti],
                "delinq_2yrs": [self.delinq_2yrs],
                "inq_last_6mths": [self.inq_last_6mths],
                "open_acc": [self.open_acc],
                "pub_rec": [self.pub_rec],
                "revol_bal": [self.revol_bal],
                "revol_util": [self.revol_util],
                "total_acc": [self.total_acc],
                "initial_list_status": [self.initial_list_status],
                "out_prncp": [self.out_prncp],
                "out_prncp_inv": [self.out_prncp_inv],
                "total_pymnt": [self.total_pymnt],
                "total_pymnt_inv": [self.total_pymnt_inv],
                "total_rec_prncp": [self.total_rec_prncp],
                "total_rec_int": [self.total_rec_int],
                "total_rec_late_fee": [self.total_rec_late_fee],
                "recoveries": [self.recoveries],
                "collection_recovery_fee": [self.collection_recovery_fee],
                "last_pymnt_amnt": [self.last_pymnt_amnt],
                "collections_12_mths_ex_med": [self.collections_12_mths_ex_med],
                "tot_coll_amt": [self.tot_coll_amt],
                "tot_cur_bal": [self.tot_cur_bal],
                "total_rev_hi_lim": [self.total_rev_hi_lim],
                "home_ownership_ANY": [self.home_ownership_ANY],
                "home_ownership_MORTGAGE": [self.home_ownership_MORTGAGE],
                "home_ownership_NONE": [self.home_ownership_NONE],
                "home_ownership_OTHER": [self.home_ownership_OTHER],
                "home_ownership_OWN": [self.home_ownership_OWN],
                "home_ownership_RENT": [self.home_ownership_RENT],
                "purpose_car": [self.purpose_car],
                "purpose_credit_card": [self.purpose_credit_card],
                "purpose_debt_consolidation": [self.purpose_debt_consolidation],
                "purpose_educational": [self.purpose_educational],
                "purpose_home_improvement": [self.purpose_home_improvement],
                "purpose_house": [self.purpose_house],
                "purpose_major_purchase": [self.purpose_major_purchase],
                "purpose_medical": [self.purpose_medical],
                "purpose_moving": [self.purpose_moving],
                "purpose_other": [self.purpose_other],
                "purpose_renewable_energy": [self.purpose_renewable_energy],
                "purpose_small_business": [self.purpose_small_business],
                "purpose_vacation": [self.purpose_vacation],
                "purpose_wedding": [self.purpose_wedding],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)