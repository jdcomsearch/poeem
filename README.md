# Poeem

Poeem is a library for efficient approximate nearest neighbor (ANN) search, which has been widely adopted in industrial recommendation, advertising and search systems.
Apart from other libraries, such as Faiss and ScaNN, which build embedding indexes with already learned embeddings, Poeem jointly learn the embedding index together with retrieval model in order to avoid the quantization distortion.
Consequentially, Poeem is proved to outperform the previous methods significantly, as shown in our SIGIR paper.
Poeem is written based on Tensorflow GPU version 1.15, and some of the core functionalities are written in C++, as custom TensorFlow ops. 
It is developed by JD.com Search.

For more details, check out our SIGIR 2021 paper [here](https://arxiv.org/abs/2105.03933).

## Content
*  [System Requirements](#system_requirements)
*  [Quick Start](#quick_start)
*  [Authors](#authors)
*  [How to Cite](#cite)
*  [License](#license)

## <a name="system"></a> System Requirements
- We only support Linux systems for now, e.g., CentOS and Ubuntu. Windows users might need to build the library from source.
- Python 3.6 installation.
- TensorFlow GPU version 1.15 (pip install tensorflow-gpu==1.15.0). Other TensorFlow versions are not tested.
- CUDA toolkit 10.1, required by TensorFlow GPU 1.15.



## <a name="quick_start"></a> Quick Start

Poeem aims at an almost drop-in utility for training and serving large scale embedding retrieval models. We try to make it easy to use as much as we can.

### Install
Install poeem for most Linux system can be done easily with pip.
```shell
$ pip install poeem
```

### Quick usage
As an extreme simple example, you can use Poeem simply by the following commands
```python
>>> import tensorflow as tf, poeem
>>> hparams = poeem.embedding.PoeemHparam()
>>> poeem_indexing_layer = poeem.embedding.PoeemEmbed(64, hparams)
>>> emb = tf.random.normal([100, 64])  # original embedding before indexing layer
>>> emb_quantized, coarse_code, code, regularizer = poeem_indexing_layer.forward(emb)
>>> emb = emb - tf.stop_gradient(emb - emb_quantized)   # use this embedding for downstream computation
>>> with tf.Session() as sess:
>>>   sess.run(tf.global_variables_initializer())
>>>   sess.run(emb)
```

### Tutorial
The above simple example, as a quick start, does not show how to build embedding index and how to serve it online. 
Experienced or advanced users who are interested in applying it in real-world or industrial system, can further read the tutorials.

- [Synthetic Data Tutorial](https://github.com/jdcomsearch/poeem/blob/master/notebook/synthetic_data_tutorial.ipynb)
- [MovieLen Tutorial](https://github.com/jdcomsearch/poeem/blob/master/notebook/movie_len.ipynb)


## <a name="authors"></a> Authors

The main authors of Poeem are:

- [Han Zhang](https://lonway.github.io/) wrote most Python models and conducted most of experiments.
- [Hongwei Shen](https://www.linkedin.com/in/hongwei-shen-27171a32/) wrote most of the C++ TensorFlow ops and managed the pip released package.
- [Yunjiang Jiang](https://www.linkedin.com/in/yunjiang-jiang-1ba96071/) developed the rotation algorithm and wrote the related code.
- [Wen-Yun Yang](https://www.linkedin.com/in/wen-yun-yang-31b48740/) initiated the Poeem project, wrote some of TensorFlow ops, integrated different parts and wrote the tutorials.


## <a name="cite"></a> How to Cite

Reference to cite if you use Poeem in a research paper or in a real-world system

```
  @inproceeding{poeem_sigir21,
    title={Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index},
    author={Han Zhang, Hongwei Shen, Yiming Qiu, Yunjiang Jiang, Songlin Wang, Sulong Xu, Yun Xiao, Bo Long and Wen-Yun Yang},
    booktitle={The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={},
    year={2021}
}
```

## License

[MIT licensed](https://github.com/jdcomsearch/poeem/blob/main/LICENSE)

