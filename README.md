# Deep Learning-based Ensemble Method for Temporal Knowledge Graph Embedding in Link Prediction.

**Three experiments are conducted in this project.**
- Each individual TKGE model is re-predicted based on the same dataset as other experiments, namely DE-SimplE, DE-DistMult, DE-TransE, ATiSE, and TERO.
- To demonstrate that the neural network can learn better weights than other methods, conducted the grid search experiment as the baseline method.
- Conducted two neural network experiments with different numbers of inputs. For 5 inputs, it took the simulated score of the top 1 prediction for each TKGE model, and for 25 inputs, it took the simulated score of the top 5 predictions for each TKGE model.

**TKGE Models**
- DE-SimplE (including DE-SimplE, DE-DistMult, and DE-TransE)
- TERO
- ATiSE

**TKGE Dataset**
- ICEWS14

**Neural Network Models**
- `25input`
- `5input`
  
**Neural Network Dataset**
- `25t_5w` has 25 inputs corresponding to the top 5 predictions of each TKGE model, and target values are the rank of the correct answer of each TKGE model.
- `5p_5w` has 5 inputs corresponding to 5 TKGE models, 5 predictions, and target values are the rank of the correct answer of each TKGE model.

