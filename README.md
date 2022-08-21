# EmoMV: Affective Music-Video Correspondence Learning Datasets for Classification and Retrieval

The goal of this project is to construct a collection of three datasets (called EmoMV) for affective correspondence learning between music and video modalities. The first two datasets (called EmoMV-A, and EmoMV-B, respectively) are constructed by making use of music video segments from other available datasets. The third one called EmoMV-C is created from music videos that we self-collected from YouTube. The music-video pairs in our datasets are annotated as matched or mismatched in terms of the emotions they are conveying. The emotions are annotated by humans in the EmoMV-A dataset, while in the EmoMV-B and EmoMV-C datasets they are predicted using a pretrained deep neural network. A user study is carried out to evaluate the accuracy of the “matched” and “mismatched” labels offered in the EmoMV dataset collection. In addition to creating three new datasets, a benchmark deep neural network model for binary affective music-video correspondence classification is also proposed. This proposed benchmark model is then modified to adapt to affective music-video retrieval.  

## Datasets: 

In folders DS1-EmoMV-A, DS2-EmoMV-B and DS3-EmoMV-C, only the extracted audio and visual features together with annotations are available. 
To download the whole dataset collection (including music-video segments together with their annotations and the extracted audio and visual features), please click [here](https://doi.org/10.5281/zenodo.7011072)

## Prerequisites

The code snippets are for binary affective music-video correspondence classification and affective music-video retrieval. They were implemented in Ubuntu 18, Python 3.6, and experiments were run on a NVIDIA GTX 1070. 

## Paper & Citation

If you use this dataset collection and/or code, please cite the following paper: 

@article{thao4189323emomv,
  title={Emomv: Affective Music-Video Correspondence Learning Datasets for Classification and Retrieval},
  author={Thao, Ha Thi Phuong and Herremans, Dorien and Roig, Gemma},
  journal={Available at SSRN 4189323}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


