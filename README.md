This is the repo for [Two Counterexamples to *Tokenization and the Noiseless Channel*](https://aclanthology.org/2024.lrec-main.1469/), presented at LREC-COLING 2024.

The poster can be found in this repo, and a recording of our presentation can be found [here](https://youtu.be/eXIZHS5_7G4).

To run an experiment, do the following things:
1. Clone the repo
2. Navigate to the `fairseq` directory
3. Install `fairseq` as suggested in that README (that is, make sure you have all the dependencies and then run the `pip install --editable ./` command)
4. Navigate to `examples/duplication_bpe` or `examples/random_drop_bpe`
5. Run the `train-duplication-bpe.sh` or `train-random-drop.sh` scripts (follow the examples below)
6. The results will appear in the `fairseq/experimental_outputs/<EXPERIMENT_NAME>` directory.


Examples for running the scripts:

```
bash train-duplication-bpe.sh --experiment-name "duplication_example" --src-bpe-tokens 4000 --tgt-bpe-tokens 4000 --duplication-n 100 --duplication-k 3 --seed 100 --device 0
bash train-random-drop.sh --experiment-name "random_drop_example" --src-bpe-tokens 6000 --tgt-bpe-tokens 6000 --random-drop-n 2000 --random-drop-k 1000 --seed 100 --bpe-seed 0 --device 0
```
