#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request,json,jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('SVR_Boston_house_prive_prediction.pkl','rb'))

@app.route('/')
def predict():
    house_price = str(model.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2,242,17.8,396.90,9.14]]))
    return house_price
                                 
if __name__ == '__main__':
    app.run(debug=True)

