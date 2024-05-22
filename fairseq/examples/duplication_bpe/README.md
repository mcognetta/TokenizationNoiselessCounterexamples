An example invocation is: `bash train-duplication-bpe.sh --experiment-name "duplication_example" --src-bpe-tokens 4000 --tgt-bpe-tokens 4000 --duplication-n 100 --duplication-k 3 --seed 100 --device 0`

No modification of the script should be necessary. The output will be in `fairseq/experiment_outputs/<EXPERIMENT_NAME>` (the experiment name is slightly different than what you put into that parameter, it is a concatenation of all the parameters so you can uniquely identify it).