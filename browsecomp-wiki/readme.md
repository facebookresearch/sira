# BrowseComp-Wikipedia

BrowseComp-Wikipedia is a 232-question benchmark for evaluating retrieval over English Wikipedia, distilled from [BrowseComp-Plus](https://github.com/texttron/browsecomp-plus). This directory contains the dataset and the pipeline we use to download, preprocess, and index the Wikipedia corpus it runs against.

## Dataset

BrowseComp-Wikipedia is the subset of BrowseComp-Plus questions that are answerable from English Wikipedia alone. We build it in three steps:

1. **Filter by source.** We keep only questions whose gold document labels include at least one English Wikipedia page, and restrict each question's gold documents to those Wikipedia pages.
2. **Verify answerability.** We give an LLM (Claude Opus 4.6) each question together with its gold Wikipedia pages and ask it to answer. A question is kept only if the model answers correctly — evidence that it is answerable from Wikipedia alone.
3. **Attach categories.** For each gold page we attach its Wikipedia categories, pulled from the dump (see [Download and Preprocess Wikipedia](#download-and-preprocess-wikipedia)).

This yields the final 232 questions.

## Results

Gold-page recall on BrowseComp-Wikipedia (all values are percentages). The benchmark contains 232 BrowseComp queries answerable from English Wikipedia, retrieved over a corpus of 25,587,229 indexed Wikipedia documents. The Perplexity baseline browses `en.wikipedia.org` interactively for up to 100 turns; SIRA must issue a single corpus-grounded BM25 query before reading any retrieved page. SIRA uses no index-time LLM enrichment here — instead it uses Wikipedia categories as corpus-side enrichment and validates proposed category tokens against the Wikipedia category graph.

| System | Backbone LLM | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|---|
| Perplexity agent | Claude 4.6 Opus | 2.59 | 4.74 | 32.33 |
| Perplexity agent | GPT-5.4 | 3.02 | 6.90 | 31.47 |
| SIRA | Claude 4.6 Opus | 9.70 | 15.27 | 36.14 |
| SIRA | GPT-5.4 | 5.71 | 13.13 | 18.51 |

## Download and Preprocess Wikipedia

We mirror the monthly English Wikipedia dump and turn it into a set of clean, query-ready parquet files. We download the article text along with Wikipedia's category-link tables, parse them into a unified page table, and derive category information on top of it — each page's categories and where those categories sit in the category hierarchy. Finally we denormalize everything into a single page-centric file, `pages_with_categories.parquet`, with one row per page carrying its title, short description, categories, and body text. The pipeline is resumable: each step skips work whose output already exists.

## Index Wikipedia

We build a BM25X sparse index directly over `pages_with_categories.parquet` using a bigram configuration (a mix of unigram and bigram scoring). We tokenize the title, description, and body text normally, and index each page's categories in two complementary ways: the categories are folded into the text as plain surface words that go through normal tokenization, and each full category is additionally hashed into a single atomic token that survives the tokenizer intact rather than being shredded. Because indexing and query time share the same normalization and hashing, this enables exact matching on categories.
