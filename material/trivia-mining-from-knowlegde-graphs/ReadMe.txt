***************************************************************************************************************
DBpedia Trivia Miner (Dataset) 
***************************************************************************************************************

We are making the datasets we used in our paper public. 
There are four files in the "Dataset" folder:

i) Bollywood_Actors.xlsx  - Bollywood Actors
ii) Music_Artists.xlsx - Music Artists
iii) Bollywood_Actors_DBpedia_Mapfile.xlsx
iv) Music_Artists_DBpedia_Mapfile.xlsx

** Open Bollywood_Actors.xlsx or Music_Artists.xlsx :

1. The first column shows the Interestingness label on a two-level scale: 0 - Boring and 1 - Interesting. These labels are the majority of five individual judgments given by five annotators taken as our gold class label .

2. The second column shows the triple fact (entity, relation/predicate, object) from DBpedia . 

3. The third column was the sentence which was shown to the annotators describing the fact triple. The sentence was generated using simple rules.

** Open Bollywood_Actors_DBpedia_Mapfile.xlsx or Music_Artists_DBpedia_Mapfile.xlsx :

These are the mapping files which maps each entity/object (first column) mentioned in files i) and ii) to its full DBpedia URL (second column).


** There are two files in the "Guidelines" folder. These guidelines were given to the human annotators to guide them on how to do the markings for the task.

