This folder runs the simulations in the CARLO environment 

For training the g_tilde in our approach run:
python test_ours.py --train 

If no other arguments are provided, the code will run with default values for the hyperparameters
To change the hyperparameters, you can go to test_ours.py and see the arguments that are available.

TO evaluate our approach using known prior:
python test_ours.py --eval --noise <noise> --bias <bias>

This command will run the evaluation for all approaches (ours and baselines) for the given noise and bias values.
To evaluate our approach using uniform prior use the '--uniform' argument.

