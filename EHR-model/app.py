import pickle
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


def greet(
    gender,
    age,
    smoking,
    pp,
    allergy,
    wheezing,
    alcohol,
    cp,
    bp,
    sugar,
    bu,
    na,
    k,
    hb,
    rbc,
    wbc,
    htn,
    appet,
    ane,
    hd,
    bmi,
    chol,
    ca,
    thal,
    insulin,
):
    with open("nvd.pkl", "rb") as f:
        nvd = pickle.load(f)
    with open("kcd.pkl", "rb") as f:
        kcd = pickle.load(f)
    with open("mp.pkl", "rb") as f:
        mp = pickle.load(f)
    with open("osp.pkl", "rb") as f:
        osp = pickle.load(f)
    with open("sk.pkl", "rb") as f:
        sk = pickle.load(f)
    lung = nvd.predict(
        np.array([[gender, age, smoking, pp, allergy, wheezing, alcohol, cp]])
    )
    kid = kcd.predict(
        np.array([[age, bp, sugar, bu, na, k, hb, wbc, rbc, htn, appet, ane]])
    )
    bra = mp.predict(np.array([[gender, age, htn, hd, sugar, bmi, smoking]]))
    hrt = osp.predict(np.array([[age, gender, cp, chol, ca, thal]]))
    dia = sk.predict(np.array([[sugar, bp, insulin, bmi, age]]))
    return [str(lung[0]), str(kid[0]), str(bra[0]), str(hrt[0]), str(dia[0])]


iface = gr.Interface(
    fn=greet,
    inputs=[
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
    ],
    outputs=["number", "number", "number", "number", "number"],
)
iface.launch(share=True)