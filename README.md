# T3

This is the official code base for the paper, "T3: Tree-Autoencoder Constrained Adversarial Text Generation for Targeted Attack".

This repo contains the code to attack both classification models (self-attentive models and BERT) and question answering models (BiDAF and BERT). We put our attack code in each folders. 

You may use our code to attack other NLP tasks.

## Note

Before using our T3(Sent), a tree-based autoencoder needs to be trained in a large corpus.

### Train Tree-based Autoencoder

We trained our tree-based autoencoder on the Yelp review training dataset. 

Related code can be found `SAM-attack/my_generator/`. Before training, each 
sentence in the training set should be parsed by Stanford CoreNLP Parser to get its dependency structures.

## Contributions

We welcome all kinds of contribution by opening a pull request. If you have questions, please open an issue for discussion.   

