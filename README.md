# Deepmind object tracking challenge
Deepmind object tracking [challenge](https://ptchallenge-workshop.github.io/)

### Quick Start

```bash
make build
make run
```

### Useful links

* [Challenge website](https://ptchallenge-workshop.github.io/)
* [Announcement blog post](https://www.deepmind.com/blog/measuring-perception-in-ai-models)
* [eval.ai](https://eval.ai/web/challenges/challenge-page/2094/overview) challenge page
* [Official paper arXiv:2305.13786](https://arxiv.org/pdf/2305.13786.pdf)
* [Official Github reference](https://github.com/deepmind/perception_test)


## Repo structure

* `data` - Data, including `annot` folder with jsons, `datasets` with prepared datasets for training, `experiments` - saved models.
* `scripts` - Small, stand alone scripts.
* `src` - The main project code location. E.g., object detection model code is in `detect` subfolder.
* `notebooks` - Jupyter notebooks.