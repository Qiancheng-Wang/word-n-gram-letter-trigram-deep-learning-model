Developed for my honour project [ANU Advanced Computing Honour Project 2017]

1. Code based on [Deep-Semantic-Similarity-Model]
2. Model defined in [Word-g-gram & Letter-trigram CLSM] 

   [ANU Advanced Computing Honour Project 2017]: <https://github.com/WrynnWang/Empirical-study-Russian-Stackoverflow>

   [Deep-Semantic-Similarity-Model]: <https://github.com/airalcorn2/Deep-Semantic-Similarity-Model>
   

   [Word-g-gram & Letter-trigram CLSM]: <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf>


- trigram_generate: generate all the letter_trigram s from the corpus that we use to train and valid our clsm model.
- read_trigram_list : read the letter_trigram s from the generated files.
- firststep: train the model (the json file include query pair and tags for both question titles)
- load_evaluate: 
- rank_at_k: experiments. rank@1 rank@5 rank@10 MAP
- russian_rank : get some good examples for the last part of our paper. (the json file we need manually generate from database, the queries here are all selected by ourself)






Libraries used in the project are:
numpy
keras
nltk
json
random
