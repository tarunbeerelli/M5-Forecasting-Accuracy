import pandas as pd
import numpy as np
import pickle

calendar = pd.read_csv("calendar.csv")
sales_data = pd.read_csv("sales_train_evaluation.csv")
sell_prices = pd.read_csv("sell_prices.csv")

infile = open('encoding','rb')
encoding_dict = pickle.load(infile)
infile.close()

infile = open('best_models','rb')
best_model = pickle.load(infile)
infile.close()

def final_1(product,store):
  x = product + str("_") + store + str("_evaluation")
  x_sales_data = sales_data[sales_data.id==x].reset_index(drop=True)
  for i in range(1942,1970):
    x_sales_data["d_"+str(i)]=np.nan
  x_data = pd.melt(x_sales_data,id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],var_name='day_no')
    
  x_sell_prices = sell_prices[(sell_prices.store_id==store) & (sell_prices.item_id==product)]
  calendar_data = calendar.rename(columns={'d': 'day_no'})
  calendar_data['event_name_1'].fillna("no_event",inplace=True)
  calendar_data['event_type_1'].fillna("no_event",inplace=True)
  calendar_data['event_name_2'].fillna("no_event",inplace=True)
  calendar_data['event_type_2'].fillna("no_event",inplace=True)

  ds=[]
  for i in calendar_data.date:
    m=i.split("-")[2]
    ds.append(m)
  calendar_data.date=ds

  x_data = pd.merge(x_data,calendar_data,how='left',on='day_no')
  x_data.day_no = x_data['day_no'].apply(lambda x: x.split('_')[1]).astype(np.int16)
  x_data = pd.merge(x_data,x_sell_prices,on=['store_id', 'item_id', 'wm_yr_wk'],how='left')
  temp = x_data[['id','day_no','sell_price']]
  temp = temp.pivot(index='id',columns='day_no',values='sell_price')
  temp.interpolate(limit_direction='backward',inplace=True,axis=1)
  temp = pd.melt(temp,ignore_index=False,value_name='sell_price')
  temp = temp.reset_index(level=0)
  x_data['imputed']=x_data['sell_price'].isna()
  x_data = x_data.drop(['sell_price'],axis=1)
  x_data = pd.merge(x_data,temp,on=['id','day_no'])

  state = store.split("_")[0]
  temp = ["CA","TX","WI"]
  temp = [i for i in temp if i!=state]
  x_data.rename(columns = {'snap_'+str(state):'SNAP'}, inplace = True)
  x_data = x_data.drop(["snap_"+str(temp[0]),"snap_"+str(temp[1])],axis=1)

  x_data=x_data.drop("weekday",axis=1)
  x_data["is_Weekend"]=(x_data.wday<2).astype("bool")
  x_data["SNAP"] = x_data["SNAP"].astype('bool')

  shiftt = 28
  for i in range(6):
    x_data["lag_"+str(shiftt+(7*i))] = x_data.groupby("id")["value"].shift(shiftt+(7*i)).fillna(0).astype(np.int16)

  for i in range(1,7):
    x_data["rolling_"+str(7*i)] = x_data.groupby("id")["lag_28"].transform(lambda s: s.rolling(i).mean())

  x_data = x_data[(x_data.day_no>=1069)]

  temp = x_data[["id","sell_price","value"]][(x_data.day_no<1914)].copy()
  temp['revenue']=(temp.value*temp.sell_price).values
  temp=temp.groupby(["id"])["revenue"].sum()
  temp=pd.DataFrame(temp)
  temp.reset_index(inplace=True)
  x_data=pd.merge(x_data,temp,on='id')

  temp=x_data[(x_data.day_no<1914)][["id","wday","state_id","SNAP","value"]].copy()
  temp.rename(columns = {'value':'trend_value'}, inplace = True)
  temp=temp.groupby(["id","wday","SNAP","state_id"])["trend_value"].mean()
  temp=pd.DataFrame(temp)
  temp.reset_index(inplace=True)
  x_data=pd.merge(x_data,temp,on=["id","wday","SNAP","state_id"])

  x_data = x_data[(x_data.day_no>=1942)]

  cat_cols = ['id','item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'date', 'wday', 'month',
            'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'SNAP']
  for col in cat_cols:
      x_data[col] = x_data[col].astype('category')
      lenc = encoding_dict[col]
      x_data[col] = lenc.transform(x_data[col].astype(str)).astype(np.int16)

  icols = x_data.select_dtypes('integer').columns
  fcols = x_data.select_dtypes('float').columns
  x_data[icols] = x_data[icols].apply(pd.to_numeric, downcast='integer')
  x_data[fcols] = x_data[fcols].apply(pd.to_numeric, downcast='float')

  x_data = x_data.drop(['value'],axis=1)

  states = {"CA":0 , "TX":1 , "WI":2}
  lgb_reg_best = best_model[states[state]]

  custom_predictions = pd.DataFrame()
  custom_predictions["id"] = x_sales_data["id"].copy()
  for a,l in enumerate(range(1942,1970)):
        custom_predictions['F'+str(a+1)] = np.round(lgb_reg_best.predict(x_data[x_data['day_no'] == l]))

  return custom_predictions
         
from flask import Flask,render_template,request,render_template_string,abort,jsonify 
app = Flask(__name__) 
   
@app.route("/")
def hello():
    return "Hello World!"

@app.route('/index')
def index():
    return render_template('index.html')

@app.errorhandler(406)
def not_found(e):
    return jsonify(error = str(e)), 404

def form_error(product,store):
    x = product + str("_") + store + str("_evaluation")
    if x not in sales_data.id.values:
        abort(406,"Please check the product id ; invalid product_id")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        product = request.form['product_id']
        store = request.form['store_id']
        form_error(product,store)

        predictions = final_1(product,store)
        predictions_html = predictions.to_html()
        return render_template_string(predictions_html)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)