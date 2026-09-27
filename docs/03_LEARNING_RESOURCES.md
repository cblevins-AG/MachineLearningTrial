# 03 · Learning resources

A curated, tiered list for an experienced developer new to Python and ML. Each track runs from
**beginner** (get productive this week) to **intermediate** (understand what you're tuning) to
**advanced** (read the papers behind the models this repo uses). Everything listed is free
unless marked 💲. Links were checked on 27 September 2026.

How to use this list: don't read it all up front. Do the ⭐ items in the beginner tier alongside
[01_GETTING_STARTED.md](01_GETTING_STARTED.md), then pick from the higher tiers when a concrete
question comes up ("why is my model over-fitting?", "what does SSIM actually measure?").

---

## Track A: Python and the data-science stack

### Beginner
* ⭐ **[The Python Tutorial](https://docs.python.org/3/tutorial/)** (official). Sections 3–9
  cover everything this repo uses. An experienced developer can skim it in an afternoon.
* **[Python in VS Code tutorial](https://code.visualstudio.com/docs/python/python-tutorial)**
  and **[Jupyter notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)**
  (Microsoft): the editor workflow, interpreter selection and debugging.
* **[Python virtual environments: a primer](https://realpython.com/python-virtual-environments-a-primer/)**
  (Real Python): what `.venv` really is and why it exists.
* ⭐ **[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)**
  (official): arrays, shapes, slicing, broadcasting. Images *are* NumPy arrays.
* **[10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)** (official):
  enough to work with the label CSVs.

### Intermediate
* **[Kaggle Learn: Python, Pandas, Data Visualization](https://www.kaggle.com/learn)**: short,
  hands-on micro-courses with in-browser exercises.
* **[pytest documentation: Get started](https://docs.pytest.org/en/stable/getting-started.html)**
  and **[fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)**: how this repo's
  tests are structured (`conftest.py` fixtures ≈ xUnit class fixtures).
* **[Python type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)**
  (mypy docs): the annotation syntax you'll see everywhere.

### Advanced
* **[Fluent Python, 2nd ed.](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)**
  (Luciano Ramalho, O'Reilly) 💲: idiomatic Python for people who already know another language well.
* **[From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)**
  (Nicolas Rougier): vectorisation thinking; why loops are replaced by array operations.

## Track B: Machine learning and deep learning fundamentals

### Beginner
* ⭐ **[Practical Deep Learning for Coders](https://course.fast.ai/)** (fast.ai): top-down and
  code-first, written for programmers. Lesson 1 trains an image classifier with transfer
  learning, which is exactly Project 1's approach. The best single starting point.
* ⭐ **[Neural networks series](https://www.3blue1brown.com/topics/neural-networks)**
  (3Blue1Brown): visual intuition for what a network, gradient descent and backpropagation
  actually are. About an hour of video.
* **[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)**
  (Google): concise modules on loss, generalisation, over-fitting, train/validation/test splits
  and classification metrics.
* **[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)**
  (PyTorch): tensors, datasets and dataloaders, building a model, the training loop, saving and
  loading. Maps one-to-one onto `train.py`.

### Intermediate
* ⭐ **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)**
  (Andrej Karpathy): the most practical essay on debugging training: look at the data, over-fit
  a small batch first, add complexity gradually. Read it before your first real training run.
* **[Dive into Deep Learning](https://d2l.ai/)** (Zhang, Lipton, Li, Smola): free interactive
  book with PyTorch code. Chapters on CNNs, modern CNNs (ResNet and its successors) and
  fine-tuning.
* **[Hands-On Machine Learning with Scikit-Learn and PyTorch](https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/)**
  (Aurélien Géron, O'Reilly, 2025) 💲: the PyTorch edition of the classic end-to-end ML book.
* **[scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)**
  and **[Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)**:
  data leakage and grouped splits, the reasons behind `split.py`.
* **[Introduction to Data-Centric AI](https://dcai.csail.mit.edu/)** (MIT): finding label
  errors, dataset curation and class imbalance. With small datasets, this is where accuracy
  comes from.

### Advanced
* **[Deep Learning](https://www.deeplearningbook.org/)** (Goodfellow, Bengio, Courville): the
  theory reference; read specific chapters as needed (optimisation, regularisation).
* **[Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)**
  (Chip Huyen, O'Reilly) 💲: data pipelines, evaluation, deployment and monitoring. The
  "production engineering" view you'll recognise from your day job.
* **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)**
  (Google): hard-won engineering practices for shipping ML.

## Track C: Image classification (Project 1)

### Beginner
* ⭐ **[Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)**
  (PyTorch tutorial): fine-tuning versus using a frozen feature extractor, in about 100 lines.
* **[Transforming and augmenting images](https://docs.pytorch.org/vision/stable/transforms.html)**
  (torchvision docs): the augmentation API used in `dataset.py`.
* **[Kaggle Learn: Computer Vision](https://www.kaggle.com/learn/computer-vision)**: CNNs,
  convolution and pooling, and transfer learning with exercises.

### Intermediate
* **[CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/)** (Stanford course
  notes). Especially "Transfer Learning" and "Neural Networks Part 3" (babysitting the learning
  process, reading loss curves).
* **[timm documentation](https://huggingface.co/docs/timm)**: choosing and loading pretrained
  backbones, data configs and fine-tuning tips; the library behind `model.py`.
* **[Hugging Face Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)**:
  classification, detection and segmentation with modern models, including zero-shot vision.
* **[Zero-shot object detection guide](https://huggingface.co/docs/transformers/tasks/zero_shot_object_detection)**
  (Hugging Face): how OWL-ViT/OWLv2 find objects from a text prompt, as used by `localize.py`.
* **[Label Studio documentation](https://labelstud.io/guide/)**: labelling bounding boxes if you
  graduate from zero-shot to a trained detector.

### Advanced (the papers behind this repo's models)
* **How transferable are features in deep neural networks?** (Yosinski et al., 2014):
  [arXiv:1411.1792](https://arxiv.org/abs/1411.1792). Why transfer learning works.
* **A ConvNet for the 2020s** (ConvNeXt; Liu et al., 2022): [arXiv:2201.03545](https://arxiv.org/abs/2201.03545). The default backbone.
* **DINOv2: Learning Robust Visual Features without Supervision** (Oquab et al., 2023):
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). The linear-probe backbone.
* **Scaling Open-Vocabulary Object Detection** (OWLv2; Minderer et al., 2023):
  [arXiv:2306.09683](https://arxiv.org/abs/2306.09683). The ribbon localiser.
* **SAM 2: Segment Anything in Images and Videos** (Ravi et al., 2024):
  [arXiv:2408.00714](https://arxiv.org/abs/2408.00714). The next step if backgrounds leak into predictions.
* **Fine-Grained Image Analysis with Deep Learning: A Survey** (Wei et al., 2021):
  [arXiv:2111.06119](https://arxiv.org/abs/2111.06119). Telling similar folds apart is a
  *fine-grained* problem; this surveys the techniques.

## Track D: Image similarity and comparison (Project 2)

### Beginner
* ⭐ **[Structural similarity index](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html)**
  (scikit-image example): why SSIM tracks perceived similarity better than raw pixel error.
* **[OpenCV-Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)**
  (official): "Image Processing in OpenCV" covers colour spaces, thresholding, morphology
  (dilation) and contours/connected components, all used in `compare.py`. OpenCV 5 keeps these APIs.
* **[Color difference](https://en.wikipedia.org/wiki/Color_difference)** (Wikipedia): CIELAB and
  delta E (CIE76 → CIEDE2000), the basis of `color_tolerance`.

### Intermediate
* **[pikepdf documentation](https://pikepdf.readthedocs.io/)** and
  **[pypdfium2 documentation](https://pypdfium2.readthedocs.io/)**: the PDF structure editor and
  renderer used in `pdf_layers.py`.
* **ISO 32000-2 (PDF 2.0)**, section 8.11 "Optional content": the precise definition of layers
  (OCGs), their default configuration and membership dictionaries. Free (registration
  required) via the PDF Association's [sponsored standards](https://pdfa.org/sponsored-standards/) page.
* **[Claude vision documentation](https://platform.claude.com/docs/en/build-with-claude/vision)**:
  how images are sent to the model, sizing and limits; the basis of the optional `ClaudeDescriber`.

### Advanced
* **Image Quality Assessment: From Error Visibility to Structural Similarity** (Wang, Bovik,
  Sheikh, Simoncelli, 2004), the SSIM paper:
  [PDF from the authors' lab](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf).
* **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (LPIPS; Zhang et
  al., 2018): [arXiv:1801.03924](https://arxiv.org/abs/1801.03924). Deep-feature similarity:
  what you would reach for if inputs became photos instead of vector renders.
* **Learning Transferable Visual Models From Natural Language Supervision** (CLIP; Radford et
  al., 2021): [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Image embeddings for
  "semantic" similarity, and the foundation OWLv2 builds on.

## Track E: From notebook to production (.NET-friendly)

* **[Export a PyTorch model to ONNX](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)** (PyTorch tutorial).
* **[ONNX Runtime with C#](https://onnxruntime.ai/docs/get-started/with-csharp.html)**: run the
  exported model from .NET (ASP.NET Core API → Blazor front end).
* **[MLflow](https://mlflow.org/docs/latest/)**: experiment tracking and a model registry once
  you're comparing many runs (the `models/<run>/meta.json` files are a lightweight version).
* **[DVC](https://dvc.org/doc)**: version datasets and models like code (see
  [01_GETTING_STARTED.md](01_GETTING_STARTED.md), section 8).
* **[Made With ML](https://madewithml.com/)**: a free MLOps course covering testing, CI/CD and
  serving for ML systems.
