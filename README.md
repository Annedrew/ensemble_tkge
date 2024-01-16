# Deep Learning-based Ensemble Method for Temporal Knowledge Graph Embedding in Link Prediction.

**TKGE Models**
- DE-SimplE (including DE-SimplE, DE-DistMult, and DE-TransE)
- TERO
- ATiSE

**TKGE Dataset**
- ICEWS14

**Neural Network Models**
- `25input`
- `5input`
  
**Neural NetworkDataset**
- `25t_5w` has 25 inputs corresponding to the top 5 predictions of each TKGE model, and target values are the rank of the correct answer of each TKGE model.
- `5p_5w` has 5 inputs corresponding to 5 TKGE models, 5 predictions, and target values are the rank of the correct answer of each TKGE model.
