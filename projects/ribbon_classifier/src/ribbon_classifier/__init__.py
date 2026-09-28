"""ribbon_classifier: find the ribbon in a photo and classify its fold/cut technique.

The pipeline, one ``ribbon`` sub-command per step (see ``ribbon --help``)::

    extract-reference  reference PDF          -> candidate images + page text to label
    moa-labels         MOA YAML + images      -> labels from a deck that lists assembly steps
    manifest           labelled folders       -> data/labels.csv
    split              labels.csv             -> data/splits.csv (train / val / test)
    crop               splits.csv + photos    -> ribbon-only crops (zero-shot detector)
    probe              splits.csv             -> quick baseline accuracy (frozen features)
    train              splits.csv             -> models/<run>/model.safetensors
    evaluate           models/<run>           -> test-set report + confusion matrix
    predict            models/<run> + photos  -> label + confidence per photo
"""

__version__ = "0.1.0"
