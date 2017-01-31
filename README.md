# Language Model RNN

This is a repository holding a language model built in torch7. It is an exercise in learning torch and the construction of RNN models.

### Layout

- `SimpleMNIST.lua`: This file holds specs for a convolutional model for classifying MNIST data. It is unrelated to the rest, but a good first exercise in torch.
- `GBWLanguageMdodel.lua`: This file holds the Language Model, and has command options for initializing it with different depths, hidden sizes, data sets, etc. This is the primary file of the project.
- `evaluation.lua`: This file generates a report on a model's loss on each dataset as well as some generation samples, toa a JSON file. It has a number of command line customizations as well.
- `DynamicView.lua`: This is a file implementinga view layer that changes sizes with each new batch to match appropriate sequence length, etc.
- `Sampler.lua`: This is a layer which takes an argmax sample over the columns of a tensor, useful for taking output from our LM and generating text. Currently only works with single sequences (no batches)
- `util.lua`: Utility functions for dataset cleaning, reducing vocab size, etc.