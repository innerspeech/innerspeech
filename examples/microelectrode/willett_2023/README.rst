Technical report
----------------

This technical report is a replication of the paper `A high-performance speech neuroprosthesis <https://www.nature.com/articles/s41586-023-06377-x>`_ on the dataset published `here <https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq>`_.

Results
-------

Baseline modified from `neural_seq_decoder <https://github.com/cffan/neural_seq_decoder>`_.

Built on top of ``neural_seq_decoder`` we added a transformer layer to achieve a better performance in `InnerSpeech (RNN-transformer 3-gram rescore) <https://eval.ai/web/challenges/challenge-page/2099/leaderboard/4944>`_.

References
----------

- Original paper: `A high-performance speech neuroprosthesis <https://www.nature.com/articles/s41586-023-06377-x>`_
- Dataset: `A high-performance speech neuroprosthesis <https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq>`_
- Official TensorFlow implementation: `speechBCI <https://github.com/fwillett/speechBCI>`_
- Official Pytorch implementation: `neural_seq_decoder <https://github.com/cffan/neural_seq_decoder>`_