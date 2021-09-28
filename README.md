#Aspect term-polarity co-extraction by mutual information maximization
A PyTorch implementation of Aspect term-polarity co-extraction by mutual information maximization

This repo contains the code and data:

In this paper,we propose a novel mutual information maximization strategy (MIM) to enhance the internal connections between tasks. Specifically, MIM jointly learns sentiment classifier and aspect boundary detector, which exploits the token-level scores of boundary detection and sentiment classification to facilitate the learning of each other:
<p>
<img src="https://raw.githubusercontent.com/cwei01/MIM/main/image/model.png" width="400" hight="300">
</p>

This framework consists of two components:

- Boundary Detection

- Sentiment Classification

Both of two components utilize [BERT](https://github.com/huggingface/pytorch-pretrained-BERT) as backbone network. The boundary detection aims to propose one or multiple candidate targets based on the probabilities of the start and end positions. The polarity classifier predicts the sentiment polarity using the span representation of the given target.

## Usage
1. Install required packages：

      Python 3.6

      [Pytorch 1.1](https://pytorch.org/)

      [Allennlp](https://allennlp.org/)

2. Download pre-train models used in the paper unzip it in the current directory

    uncased [BERT-Large](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model
3. train the MIM for aspect term-polarity co-extraction and the results are in /result like this:
```shell
python -m main.run_joint_span \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --data_dir $DATA_DIR \
  --train_file rest_total_train.txt \
  --predict_file rest_total_test.txt \
  --train_batch_size 32 \
  --output_dir out/01
```
## Detailed information

The range of parameter in this paper is as follows:

```
the learning rate:[0.1,0.01,0.001,0.0001]
the batch size:[64,128,256]
the window size:[0,1,2,3,4,5,6,7]
the weight_kl:[0.0,10^-7,10^-5,10^-3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
the num_train_epochs:[20,40,50]
```

## Acknowledgements
We sincerely thank Xin Li for releasing the [datasets](https://github.com/lixin4ever/E2E-TBSA).

