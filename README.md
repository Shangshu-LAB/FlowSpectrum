# FlowSpectrum
The repository of paper *FlowSpectrum: a concrete characterization scheme of network traffic behavior for anomaly detection*, published in WWWJ 2022. 


## File

Decomposer_semiAE.py is used to build the neural network structure based on semi-supervised AutoEncoder.

Decomposer_training.py is used for continuous training of the decomposer.

FlowSpectrum.py includes the FlowSpectrum class.

FlowSpectrum_AE_Test.py is used to test the detection result using neural network structures as decomposer.

FlowSpectrum_ML_Test.py is used to test the detection result using traditional dimensionality reduction methods as decomposer.

KDD_preprocess.py is used for preprocessing of the dataset.

## Citation
For any work related to the network traffic analysis, welcome to please cite our paper as:
```
@article{Yang2022FlowSpectrumAC,
  title={FlowSpectrum: a concrete characterization scheme of network traffic behavior for anomaly detection},
  author={L. Yang and Shaojing Fu and Xuyun Zhang and Shize Guo and Yongjun Wang and Chi Yang},
  journal={World Wide Web},
  year={2022},
  volume={25},
  pages={2139 - 2161},
  url={https://api.semanticscholar.org/CorpusID:248442654}
}

Copy
```
