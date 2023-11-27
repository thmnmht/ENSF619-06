from flask import Flask, jsonify
from flask_cors import CORS
import torch
import time
from models.Ensemble import Model as Ensemble
from models.FEDformer import Model as FEDformer
from data_provider.data_factory import data_provider
from get_args import get_args
from get_new_args import get_args as get_new_args
import requests
import pandas as pd
from utils.timefeatures import time_features
import numpy as np


app = Flask(__name__)
CORS(app)

args = get_args()
new_args = get_new_args()
path = 'checkpoints/custom_Ensemble_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0'
new_model_path = 'checkpoints/NewData_FEDformer_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_newData_0/checkpoint.pth'
best_model_path = path + '/' + 'checkpoint.pth'
model = Ensemble(args)
new_model = FEDformer(args)

model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
new_model.load_state_dict(torch.load(new_model_path, map_location=torch.device('cpu')))

# DataLoader
data_set, data_loader, scaler = data_provider(args, 'test')
new_data_set, new_data_loader, new_scaler = data_provider(new_args, 'test')

global_input_output_pairs = []

@app.route('/get-history', methods=['GET'])
def get_input():
    # Return the most recent
    if global_input_output_pairs:
        io_pair = global_input_output_pairs[-1]
        return jsonify({'history': io_pair['input'].tolist()})
    else:
        return jsonify({'error': 'Nothing available yet.'})
    

@app.route('/get-prediction', methods=['GET'])
def get_output():
    # Return the most recent
    run_inference_loop()
    if global_input_output_pairs:
        io_pair = global_input_output_pairs[-1]
        return jsonify({'prediction': io_pair['output'].tolist()})
    else:
        return jsonify({'error': 'Nothing available yet.'})
    
@app.route('/new_data_pred', methods=['GET'])
def new_data_pred():
    res = new_inference()
    if res:
        return jsonify({'history_price': res['input'].tolist(),
                        'history_time': res['input_time'].tolist(),
                        'prediction_price': res['output'].tolist()})
    else:
        return jsonify({'error': 'Nothing available yet.'})


def run_inference_loop():


    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(data_loader))
       
    batch_x = batch_x.float()
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
    pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark).detach()

    x_np = data_set.inverse_transform(batch_x[0,:,:].detach().numpy())[:,0]
    y_np = data_set.inverse_transform(batch_y[0,-args.pred_len:,:].detach().numpy())[:,0]
    pred_np = data_set.inverse_transform(pred[0,:,:].detach().numpy())[:,0]

    global_input_output_pairs.append({
        'input': x_np,
        'output': pred_np,
        'labels': y_np
    })

def new_inference():

    url = "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/latest?period_id=1HRS&limit=97"

    payload={}
    headers = {
    'Accept': 'application/json',
    'X-CoinAPI-Key': '34BC0EF3-7A49-422C-AFDA-4403D4388CAE'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    data = response.json()
    reversed_df = pd.DataFrame(data)[1:]
    df = reversed_df.iloc[::-1].reset_index(drop=True)
    # df = pd.read_csv('dataset/newBtc/hourly.csv', nrows=96)



    batch_x = new_scaler.transform(df[['price_close']].values)
    data_stamp = time_features(pd.to_datetime(df['time_close'].values))
    data_stamp = data_stamp.transpose(1, 0)

    x_mark = pd.to_datetime(df['time_close'])[-args.label_len:]
    y_mark = pd.to_datetime(df['time_close'].values)+ pd.Timedelta(hours=96)
    pred_time = np.concatenate([x_mark.values, y_mark.values])
    pred_stamp = time_features(pd.to_datetime(pred_time))
    pred_stamp = pred_stamp.transpose(1, 0)




    # decoder input
    dec_inp = torch.zeros((1, args.pred_len, 1)).float()
    dec_inp = torch.cat([torch.tensor(batch_x[-args.label_len:, :].reshape(1, -1, 1)), dec_inp], dim=1).float()
    pred = new_model(torch.tensor(batch_x.reshape(1, -1, 1)).float(), torch.tensor(data_stamp).float(), 
                 dec_inp, torch.tensor(pred_stamp).float()).detach()

    x_np = new_data_set.inverse_transform(torch.tensor(batch_x).float().detach().numpy())[:,0]
    pred_np = new_data_set.inverse_transform(pred[0,:,:].detach().numpy())[:,0]

    return {'input': x_np,
            'input_time': df['time_close'].values,
            'output': pred_np,
            }
    


    

if __name__ == '__main__':
    
    app.run()
    
