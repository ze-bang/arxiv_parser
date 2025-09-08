import pandas as pd

from arxiv_cite_forecast.features.text import TextFeaturizer
from arxiv_cite_forecast.features.meta import MetaFeaturizer


def test_text_and_meta_shapes():
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "abstract": ["foo bar", "baz qux"],
            "n_authors": [2, 3],
            "primary_cat": ["cs.LG", "cs.AI"],
            "month": [1, 2],
        }
    )
    tf = TextFeaturizer(max_features=100, n_components=8)
    tf.fit(df.title, df.abstract)
    Xtxt = tf.transform(df.title, df.abstract)
    assert Xtxt.shape == (2, 8)

    mf = MetaFeaturizer()
    mf.fit(df)
    Xcat, Xnum = mf.transform(df)
    assert Xnum.shape[0] == 2
