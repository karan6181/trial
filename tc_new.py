from automodel.text import TextClassifier
from automodel.text.data import TextClassifierData

data = TextClassifierData(data_path="mydatapath")
# provides feedback on the data
# such as
# number of classes, class:data sample ratio, suggestions for collecting more data
# this will be useful to the user to get more high quality to further improve model
#data.analyze()

constraints = ModelConstraints().time_limit(1200).model_size(500).throughput(100)

textClassifier = TextClassifer()

textClassifier.fit(data,
                   metrics = ["loss", "perplexity"],
                   model_constraints = constraints,
                   resource_config = 'localhost',
                   model_generation_policy = TextClassifierPolicy.default())

textClassifier.stats() # shows stats about train_loss, val_loss, etc.,

textClassifer.evaluate(eval_data_path) # return stats

textClassifier.save(path="./")