After several attemps and very different kind of models from VGAE until the last GIN network we weren't able to reach the baseline fixed for the Hackaton. For sure our approach was completely wrong from a 'just copy and win' point of view, cause instead of use CTRL+C and CTRL+V we started exploring all the possible models, losses, optimizator and so on and so forth. We learnt a lot but don't get the point, that's life: you cannot win and learn a the same t
ime! Now we are little angry cause we spent six days of our lifes just coding as foolish is sunny days, but maybe we just need to rest.

In a serious attempt to explain what we tried in the end, we did:
- Data Augmentation adding gaussian noise on the edges and dropping edges
- We picked GIN network from the baseline and changed the normalization layers from BatchNorm to GraphNorm
- We made many runs on Optuna to find the best possible parameters, resulting in the following params:
  -lr=0.0002(that will be tuned via the implementation of a scheduler to avoid plateaus via ReduceRLOnPlateau)
  -layers=4
  -hidden units=480
  -batch size=32-64
  -gnn type=gin
  -weight decay=0.00046
  -dropout=0.5
- We tried implementing a sorta of GCODLoss from your article (https://arxiv.org/html/2412.08419v1)
- We put some early stopping to prevent overfitting
- We changed jk and graph pooling types, the first one to max and the second one to attention
- We changed the optimizer to AdamW to implement weight decay
I also add the logs that are already in the folder:
