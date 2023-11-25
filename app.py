from flask import Flask, jsonify
import torch
import time
from models.Ensemble import Model as Ensemble
from data_provider.data_factory import data_provider
from get_args import get_args


app = Flask(__name__)

args = get_args()
path = 'checkpoints/custom_Ensemble_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0'
best_model_path = path + '/' + 'checkpoint.pth'
model = Ensemble(args)

model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))

# DataLoader
data_set, data_loader = data_provider(args, 'test')

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
    if global_input_output_pairs:
        io_pair = global_input_output_pairs[-1]
        return jsonify({'prediction': io_pair['output'].tolist()})
    else:
        return jsonify({'error': 'Nothing available yet.'})


def run_inference_loop():
    # Iterates over the DataLoader

    # batch_num = 0
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # Sleep 5 second
        time.sleep(5)
        # if i == batch_num:
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
        pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark).detach()

        x_np = data_set.inverse_transform(batch_x[i,:,:].detach().numpy())[:,0]
        y_np = data_set.inverse_transform(batch_y[i,-args.pred_len:,:].detach().numpy())[:,0]
        pred_np = data_set.inverse_transform(pred[i,:,:].detach().numpy())[:,0]

        global_input_output_pairs.append({
            'input': x_np,
            'output': pred_np,
            'labels': y_np
        })


    

if __name__ == '__main__':
    # Start the Flask server in a separate thread
    from threading import Thread
    flask_thread = Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.start()
    
    # Run the inference loop
    run_inference_loop()

    # Wait for the Flask server to end
    flask_thread.join()
