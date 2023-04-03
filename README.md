# Yuan Wang's CS 224N Default Final Project - Multitask BERT

This project discovered a few successful measures that help improve performance of the multitask minBERT model against the baseline. These measures include increasing number of epochs trained, training on additional datasets, using round robin multitask training as opposed to task-by-task training, and performing additional training with minBERT parameters fixed. Ensembling these changes has improved the overall performance of the model by close to 5% across all three tasks. It was also discovered that training on the high-volume PARA Quora dataset helps improve model performance on the two other tasks.

The poster and final report could be found in the /Results folder.

The files in this project are originally from Stanford CS 224N's starter repository: https://github.com/gpoesia/minbert-default-final-project. The code files are edited to complete this project. The most critical edits are in the the following two files: multitask_classifier.py and construct_datasets.py.

### Acknowledgement (copied from https://github.com/gpoesia/minbert-default-final-project)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
