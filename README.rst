Innerspeech: Biosignal Speech Recognition and Synthesis Toolkit
================================================================

Introduction
------------
Innerspeech is a toolkit for biosignal speech recognition and synthesis. It is designed to be used in the context of brain-computer interfaces (BCI) and other speech-related biosignal processing applications. The toolkit is written in PyTorch and PyTorch-Lightning and is designed to be easy to use and extend with the latest model architectures and datasets.

Motivation
----------
We have observed a recent trend in utilizing both invasive biosignals (e.g., Utah array, Neuralink) and non-invasive biosignals (e.g., HDEEG, EEG, EMG, fMRI, MEG) for speech recognition and synthesis. With improvements in word error rates (WER) in recent approaches (`~5.8% for invasive brain signals <https://eval.ai/web/challenges/challenge-page/2099/leaderboard/4944>`_ and `~12.2% for EMG signals <https://arxiv.org/abs/2403.05583>`_), techniques from traditional ASR/TTS/NLP are being applied to this field. However, there isn't an easy-to-use toolkit that allows researchers to benchmark various datasets, as most existing ASR/TTS toolkits assume input/output to be heavily audio-based. This makes them unsuitable for direct application to biosignal processing. Consequently, much of the work remains dispersed across separate repositories, each using different machine learning frameworks. 

Inspired by open-source speech processing toolkits like Kaldi, ESPnet, and Coqui-TTS, we aim for the Innerspeech toolkit to serve as a common ground, facilitating the synchronization of work in this field and proving useful for researchers and developers working on brain-computer interfaces (BCI) and other biosignal processing applications.

Installation
------------
Please check `installation.rst <./installation.rst>`_ for the installation of the toolkit.

Usage
-----
The main components of the toolkit are the models and training scripts on corresponding datasets. The models are implemented as PyTorch-Lightning modules which provide a simple interface for training and evaluation on multi-node multi-GPU systems. Each subfolder in ``examples`` represents one replication of a paper in this field under the naming convention of ``<modality>/<author_year>``. If the original dataset is available, we will provide a link to the original dataset and a command to download and preprocess the dataset. The toolkit includes a number of pre-implemented models with corresponding pretrained weights. In particular, the toolkit includes both supervised and unsupervised models for speech recognition and synthesis based on audio/biosignal data.

Quickstart and Tutorial
-----------------------
The following example in ``examples/microelectrode/willett_2023`` demonstrates how to train a GRU model on the `A high-performance speech neuroprosthesis <https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq>`_ dataset using the Innerspeech toolkit. This example is equivalent to the ``InnerSpeech (RNN-transformer 3-gram rescore) WER 10.08`` submission on the `Brain-to-Text Benchmark '24 <https://eval.ai/web/challenges/challenge-page/2099/leaderboard/4944>`_.

More detailed tutorials and examples are available in the documentation and the examples directory of the repository.

Benchmarking Results
--------------------
In order to keep track of the performance of the toolkit, we maintain a list of benchmarking results on a variety of datasets. The leaderboard will be made available on the `innerspeech/open-speechbci-leaderboard <https://huggingface.co/spaces/innerspeech/open-speechbci-leaderboard>`_. The benchmarking results wll be updated regularly and include the performance of the toolkit on a variety of datasets. Please feel free to contact us if you would like to add your own benchmarking results to the leaderboard.

Support
-------
If you have any questions or need help with the toolkit, please feel free to contact me at kenneth@innerspeech.ai, or open an issue on the GitHub repository.

Acknowledgements
----------------
I extend my heartfelt gratitude to Maogu, Xiaoju, Wugi, and Zihan for their unwavering support and encouragement throughout the development of this toolkit. Their contributions have been invaluable to the success of this project.

I would also like to thank the following programs for their support on funding and cloud computing credits: Nvidia Inception, AWS Activate, HKSTP Ideation, HK Tech 300, Microsoft Founder Hub, Communitech Founder Program, Google for Startups, and OVHcloud Startup Program.

Citation
--------
If you use this toolkit in your research, please cite the following paper:

.. code-block::

    @article{innerspeech,
    title={Innerspeech: Biosignal Speech Recognition and Synthesis toolkit},
    author={Wang Yau Li},
    journal={arXiv preprint arXiv:},
    year={2024},
    githbu_repo={https://github.com/innerspeech/innerspeech}
    }
