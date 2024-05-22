#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

# echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
# git clone https://github.com/rsennrich/subword-nmt.git
git clone https://github.com/mcognetta/modified-subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
# BPEROOT=subword-nmt/subword_nmt
BPEROOT=modified-subword-nmt/subword_nmt
#BPE_TOKENS=10000
SRC_BPE_TOKENS=10000
TGT_BPE_TOKENS=10000
SEED=0
DEVICE=0

EXPERIMENT_PREFIX="experiment"

while [[ "$#" -gt 0 ]]
do case $1 in
    --src-bpe-tokens) SRC_BPE_TOKENS=$2
    shift;;
    --tgt-bpe-tokens) TGT_BPE_TOKENS=$2
    shift;;
    --seed) SEED=$2
    shift;;
    --device) DEVICE=$2
    shift;;
    --experiment-name) EXPERIMENT_PREFIX="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done
echo "========= PARAMETERS =========== "
echo -e "SRC_TOKENS $SRC_BPE_TOKENS \nTGT_TOKENS $TGT_BPE_TOKENS \nSEED $SEED \nDEVICE $DEVICE \nNAME $EXPERIMENT_PREFIX"
echo "========= PARAMETERS =========== "

src=de
tgt=en
lang=de-en

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_BPE_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_seed_${SEED}.${lang}"


URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi


prep=experiments/$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $tmp/test.$l
done

BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab


# python3 $BPEROOT/learn_bpe.py --input $tmp/train.$src -s $SRC_BPE_TOKENS > $BPE_CODE.$src
# python3 $BPEROOT/learn_bpe.py --input $tmp/train.$tgt -s $TGT_BPE_TOKENS > $BPE_CODE.$tgt

python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $tmp/train.$src -s $SRC_BPE_TOKENS -t -o $BPE_CODE.$src --write-vocabulary $BPE_VOCAB.$src
python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $tmp/train.$tgt -s $TGT_BPE_TOKENS -t -o $BPE_CODE.$tgt --write-vocabulary $BPE_VOCAB.$tgt


for f in train valid test; do
    echo "apply_bpe.py ($src) to ${f}.${src}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE.$src < $tmp/$f.$src > $prep/$f.$src

    echo "apply_bpe.py ($tgt) to ${f}.${tgt}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE.$tgt < $tmp/$f.$tgt > $prep/$f.$tgt
done

cd ../..

TEXT=examples/translation/experiments/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/translation/data-bin/$EXPERIMENT_NAME \
    --workers 8 \
    --srcdict examples/translation/experiments/$EXPERIMENT_NAME/vocab.$src \
    --tgtdict examples/translation/experiments/$EXPERIMENT_NAME/vocab.$tgt


mkdir experiment_outputs/${EXPERIMENT_NAME}/

CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-train  examples/translation/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --optimizer adam --adam-betas '(0.9, 0.98)' \
                                            --clip-norm 0.0 \
                                            --lr 5e-4 \
                                            --lr-scheduler inverse_sqrt \
                                            --warmup-updates 4000 \
                                            --dropout 0.3 \
                                            --weight-decay 0.0001 \
                                            --criterion label_smoothed_cross_entropy \
                                            --label-smoothing 0.1 \
                                            --max-tokens 4096 \
                                            --eval-bleu  \
                                            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                                            --eval-bleu-detok moses \
                                            --eval-bleu-remove-bpe \
                                            --eval-bleu-print-samples \
                                            --best-checkpoint-metric bleu \
                                            --maximize-best-checkpoint-metric \
                                            --patience 5  \
                                            --save-dir "experiment_outputs/${EXPERIMENT_NAME}" \
                                            --source-lang=$src \
                                            --target-lang=$tgt \
                                            --task "translation" \
                                            --seed $SEED \
                                            --no-epoch-checkpoints > experiment_outputs/${EXPERIMENT_NAME}/$EXPERIMENT_NAME.log
                                            
CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-generate examples/translation/data-bin/$EXPERIMENT_NAME \
                                        --path experiment_outputs/${EXPERIMENT_NAME}/checkpoint_best.pt \
                                        --batch-size 128 \
                                        --beam 5 \
                                        --max-len-a 1.2 \
                                        --max-len-b 10 \
                                        --remove-bpe > experiment_outputs/${EXPERIMENT_NAME}/bleu.log