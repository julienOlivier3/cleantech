# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="G_WElanyAGHz"
# # Domain-Specific BERT Models
#
# by Chris McCormick and Nick Ryan
#

# + [markdown] id="uXKyKe3NZONV"
# # 1. Introduction

# + [markdown] id="6Laip2bqmRK-"
# If your text data is domain specific (e.g. legal, financial, academic, industry-specific) or otherwise different from the "standard" text corpus used to train BERT and other langauge models you might want to consider either continuing to train BERT with some of your text data or looking for a domain-specific language model.
#
# Faced with the issue mentioned above, a number of researchers have created their own domain-specific language models. These models are created by  training the BERT architecture *from scratch* on a domain-specific corpus rather than the general purpose English text corpus used to train the original BERT model. This leads to a model with vocabulary and word embeddings better suited than the original BERT model to domain-specific NLP problems. Some examples include: 
#
# - SciBERT (biomedical and computer science literature corpus)
# - FinBERT (financial services corpus)
# - BioBERT (biomedical literature corpus)
# - ClinicalBERT (clinical notes corpus)
# - mBERT (corpora from multiple languages)
# - patentBERT (patent corpus)
#
# In this tutorial, we will:
#
# 1. Show you how to find domain-specific BERT models and import them using the  `transformers` library in PyTorch.
# 2. Explore SciBERT and compare it's vocabulary and embeddings to those in the original BERT.
#
#
#

# + [markdown] id="ku_4hKSLCaSE"
# ## 1.1 Why not do my own pre-training?
#
# If you think your text is too domain-specific for the generic BERT, your first thought might be to train BERT from scratch on your own dataset. (Just to be clear: BERT was "Pre-Trained" by Google, and we download and "Fine-Tune" Google's pre-trained model on our own data. When I say "train BERT from scratch", I mean specifically re-doing BERT's *pre-training*).
#
# Chances are you won't be able to pre-train BERT on your own dataset, though, for the following reasons. 
#
# **1. Pre-training BERT requires a huge corpus**
#
# BERT-base is a 12-layer neural network with roughly 110 million weights. This enormous size is key to BERT's impressive performance. To train such a complex model, though, (and expect it to work) requires an enormous dataset, on the order of 1B words. Wikipedia is a suitable corpus, for example, with its ~10 million articles. For the majority of applications I assume you won't have a dataset with that many documents. 
#
# **2. Huge Model + Huge Corpus = Lots of GPUs**
#
# Pre-Training BERT is expensive. The cost of pre-training is a whole subject of discussion, and there's been a lot of work done on bringing the cost down, but a *single* pre-training experiment could easily cost you thousands of dollars in GPU or TPU time. 
#
# That's why these domain-specific pre-trained models are so interesting. Other organizations have footed the bill to produce and share these models which, while not pre-trained on your specific dataset, may at least be much closer to yours than "generic" BERT.
#

# + [markdown] id="clB4Bklb-B13"
# # 2. Using a Community-Submitted Model

# + [markdown] id="ZvcsqFp6TclZ"
# ## 2.1. Library of Models

# + [markdown] id="q3dpoWs_Dn-t"
# The list of domain-specific models in the introduction are just a few examples of the models that have been created, so it's worth looking for and trying out an open-source model in your domain if it exists.
#
# Fortunately, many of the popular models (and many unpopular ones!) have been uploaded by the community into the transformers library; you can browse the full list of models at: [https://huggingface.co/models](https://huggingface.co/models)
#
# '[![screenshot](https://drive.google.com/uc?export=view&id=1SxPIV4aXqGJxOpUw-6gkOZUEVWM5iupa)](https://drive.google.com/uc?export=view&id=1SxPIV4aXqGJxOpUw-6gkOZUEVWM5iupa)'
#
# It's not very easy to browse, however--there are currently over 1,700 models shared! 
#
# If you know the name of the model you're looking for, you can search this list by keyword. But if you're looking for a specific type of model, there is a "tags" filter option next to the search bar.
#
# For example, if you filter for "Multilingual" and "Pytorch", it narrows it done to just 10 models.
#
# [![screenshot](https://drive.google.com/uc?export=view&id=1T6nbD9DLxioW1sOdwU5tesYnHK83-USr)](https://drive.google.com/uc?export=view&id=1T6nbD9DLxioW1sOdwU5tesYnHK83-USr)
#
# > Side Note: If you skim the full list you'll see that roughly 1,000 of the current models are variants of a single machine translation model from "Helsinki NLP". There is a different variant for every pair of languages. For example, English to French: `Helsinki-NLP/opus-mt-en-fr`
#

# + [markdown] id="F2dedg7NQqga"
#
# For this Notebook, we'll use SciBERT, a popular BERT variant trained primarily on biomedical literature. 
#
# Each model has its own page in the huggingface library where you can learn a little more about it: https://huggingface.co/allenai/scibert_scivocab_uncased
#
# [![screenshot](https://drive.google.com/uc?export=view&id=1T4T0vQw9NH7WufIv1nxwQdNepozCbChs)](https://drive.google.com/uc?export=view&id=1T4T0vQw9NH7WufIv1nxwQdNepozCbChs)
#
# Here are some highlights:
# * SciBERT was trained on scientific literature--1.14M papers.
#   * ~1/5th of the papers are from "the computer science domain"
#   * ~4/5th of the papers are from "the broad biomedical domain".
# * SciBERT was created by the Allen Institute of AI (a highly respected group in NLP, if you're unfamiliar).
# * Their paper was first submitted to arXiv in March, 2019 [here](https://arxiv.org/abs/1903.10676). They uploaded their implementation to GitHub [here](https://github.com/allenai/scibert) around the same time.
#
#
#

# + [markdown] id="cFTiBsi8Ti-h"
# ## 2.2. Example Code for Importing

# + [markdown] id="aJF_caESC0yS"
# If you're interested in a BERT variant from the community models in the transformers library, importing can be incredibly simple--you just supply the name of the model as it appears in the library page.
#
# First, we'll need to install the `transformers` library.
#

# + colab={"base_uri": "https://localhost:8080/"} id="R36LrzeQGDUq" outputId="376dea5a-3976-4276-da92-49936e90caf7"
# !pip install transformers

# + [markdown] id="DIcEh7LvzMvO"
#
#
# The `transformers` library includes classes for different model architectures (e.g., `BertModel`, `AlbertModel`, `RobertaModel`, ...). With whatever model you're using, it needs to be loaded with the correct class (based on its architecture), which may not be immediately apparent. 
#
# Luckily, the `transformers` library has a solution for this, demonstrated in the following cell. These "Auto" classes will choose the correct architecture for you! 
#
# That's a nice feature, but I'd still prefer to know what I'm working with, so I'm printing out the class names (which show that SciBERT uses the original BERT classes).
#

# + colab={"base_uri": "https://localhost:8080/", "height": 218, "referenced_widgets": ["de515e7326694b62833a55098bd9564e", "c5432e7f779644f5ab6148305947979e", "948db0b6e1154bc7a878c3696a93c666", "62fc919a5c60432bb49f688294401659", "62852c6330a3457fb88a9fb53a2d1a94", "8dca32a0c2774d8197f4e318a8923175", "9ae8c332bf7049f2a11c8d32377e1604", "78bbe1d2ef914d96a20bfc5a2bec830f", "b3500798e4664daeb0f36f446554c0bb", "7c56b8d7410143839b24482394429e08", "384987b2e15f447dac744513949ac2f4", "4bf2fc1ce884456ab1f40a16f4f02dc0", "5361a997eca64720846b8b60d262b1da", "c170c7d851494ecc938ee8cb5fd927b2", "225a647f7bfa4224a87ffb8d2498855c", "ee762eb6f07b4156b54fd276fe86eddb", "ed46cbd1405743ceb99c3e8b8489adfc", "aae0b67ed889411f97eff606ff20d11b", "5520eebd0ff34b30a87669e6e4485d99", "ca6f7f2c05fc4341845d1425aa0992d6", "db2deab599154380bf738bd4210cdc9e", "ae3f16bd374d4305873be5abfa1bfe4a", "131e2fde120e4f80b0532222eb74650c", "6692cdac5401437389d65c15aaf8db83", "421f19b049a4466183f499b7d1673410", "f55ae674b7b84a21bd300f5505bc790b", "430c3a2d4eb74001869d34083598ef59", "8a9cfa9181a34e0c8984247f46983f7b", "fbc3cd3100204205a74a2504c4a22bb0", "2296b9743cfa48a9860e8bd975c3872f", "cd7837cfb9c4435eab2cec28cbecc8e6", "1237f5036c12451bb9d953e06c5598f2", "fa4b4ed6f06e46fcbddaf1d4b20c1291"]} id="oJFsRo_vGDYU" outputId="90bbb5bf-99a8-432f-8b11-aae37cd78765"
from transformers import AutoTokenizer, AutoModel
import torch

scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

print('scibert_tokenizer is type:', type(scibert_tokenizer))
print('    scibert_model is type:', type(scibert_model))


# + colab={"base_uri": "https://localhost:8080/", "height": 314, "referenced_widgets": ["5928be975bd0474891c95906f0dac531", "bfdb31b3b3124121849dacd279221049", "831645e65b2f47738c7c022ba15c3187", "5368a1b7cf9e40c0a55fffba87b1f5b4", "8ed55da5c1364b91a702e8cea84cd557", "0a833974088b4e81a56cd017e1013d3a", "a56170cade7245e0a3a4f6606e312e7b", "ef74cb0a57ed4074bcd6c9fe37591f68", "771af1f496c442c1903bf0853aaf5752", "cbb67f537e4d46119e8029dac0d30ddc", "711afac4169544a2aecfbf33551fff9c", "b9b4b522287f4bf7b26e95cffedab20c", "313bc56099774c918c18da8431e48e94", "4d5ba517075b483ab5dad20f9fdfec55", "a8493c83af794e63a739ed14ab1a29a9", "f26cb2e872744847badfe55ab3ea0da9", "ec501c3803b343bc9fc49693e84d824d", "4d0deb2d8fbc439fb8cdf8abf295794f", "337d844ced954064b016fd43541ecf4a", "4a8254756b78486199e0a2b5dc515f0e", "cb6fad174f9d479297e2fd7247c0602d", "bdbd8e56cccc40f4b47ccf7205f8f7ef", "ee4725bdb1b44977a4d1bbc79d2e47fd", "8eb84277815f431c8efb48de3c174477", "25e597ec95dd4add9127537923d2dcbb", "23e2627903d042229b8dc6efa016a9ca", "fb5a3518e49d46c08ec3ab99ae45f591", "a5a9c54a26da437e97ed4668c5864f3a", "3840983308c14620819d03851bf237ea", "d3ed8c5729914a9586d1023f7c8d61fa", "996b42da527548768e0b5c29d24ff3cc", "c3133b4b813944d1ae6d532edafe59f1", "c84ee4b080194c0b866c120d74375eac", "1b31638c61de4e6698d2578ab67ace7c", "e077a40175894460a544023e43c88d59", "7053132230cf4504b7835db6dc7e875b", "80350f6321924a0f915eff7d72baacdb", "5d3e7f0344ea4125a6958e0dad3ba0fb", "ac13500413344d02ab05e5d6741bbd64", "096579c772014e74a7cf6483e3070871", "d3eaae19d53b4b54a20a4418d9970b08", "70d864498f8e40f2a54e6a4bde65eac1", "9e9f22ac59204d44ade32d5139cf3330", "95475e03715a486f9b07dca548ce5483", "80d53672214c484da7adc34bc6866681", "9b3b3b3a17b94fa395d508bca5c4d8bd", "e8d97a1fd6554512b151f12e3d988e50", "4f3dbed7499549fabb44af4bc841819b", "2d9de4ab106244cb85e4f8e315ca21c4", "49ac12b37fb84b41bb1e0ba2a994cca0", "1e6ca3efaab843c6b8adbe39045f3344", "cd48b136f87d48998477261322908888", "082b327a3add45a383e2007f34ecd675", "8a2833e8bfc24e6ba340aaffc5ec2858", "e6df39937e6648948a25918a7517629e", "d83d3bb7d7474ac3bb53cde6a8fdd41a", "9c98dbbfcc0d43d29bfca0847ab784d6", "9728075092a74fccae5c8e83ffb17939", "c81e916b357a4af9959e3e17d48d99a7", "d2f224ff3b584a7eb11c40835a8ed34b", "a4ef0a685d8d42438ef625e89f55bfda", "17d6b9d508c94ba5a8b8d7108324b6d9", "8ead1e01498e4cf385a8c3bcf90f1f7c", "d19250deccb94b62bfd2832c2d1e10fb", "d8861e52558148f5b48ed3141ac1b54c", "b9f34491595a4f658fa9f45f84bcb7f5"]} id="zTrmnpRyMNJm" outputId="f7f86d00-cd58-4879-9fb8-c86feb47e8a3"
from transformers import AutoTokenizer, AutoModel
import torch

bigbert_tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
bigbert_model = AutoModel.from_pretrained("google/bigbird-pegasus-large-bigpatent")

print('bigbert_tokenizer is type:', type(bigbert_tokenizer))
print('    bigbert_model is type:', type(bigbert_model))

# + [markdown] id="qEVnhVwtV3d5"
# # 3. Comparing SciBERT and BERT

# + [markdown] id="uquAjkdLzg7N"
# ## 3.1. Comparing Vocabularies

# + [markdown] id="RwgdQOu_8yrC"
# The most apparent difference between SciBERT and the original BERT should be the model's vocabulary, since they were trained on such different corpuses.
#
# Both tokenizers have a 30,000 word vocabulary that was automatically built based on the most frequently seen words and subword units in their respective corpuses. 
#
# The authors of SciBERT note:
#
# > "The resulting token overlap between [BERT vocabulary] and
# [SciBERT vocabulary] is 42%, illustrating a substantial difference in frequently used words between scientific and general domain texts."
#
# Let's load the original BERT as well and do some of our own comparisons.
#
# *Side note: BERT used a "WordPiece" model for tokenization, whereas SciBERT employs a newer approach called "SentencePiece", but the difference is mostly cosmetic. I cover SentencePiece in more detail in our [ALBERT eBook](https://www.chrismccormick.ai/offers/HaABTJQH).*
#

# + colab={"base_uri": "https://localhost:8080/", "height": 248, "referenced_widgets": ["a1e5a87ead714618bcb713152ae13030", "bb57b3bcff0d4a0bb692281ce90ec600", "257af96c990c4e4e978aa1d879a89828", "ab037b5354144372a95f21310decc79f", "32b9444cd0d84776801c5ab24d28ee41", "12aaffa086fd4ea0bda86dda63d25b85", "c07667db6a204163a93d60069f7ab4d4", "1b722c5d980f4d56874f103fa1641b68", "b41cd080938a4f78b067b31e2d1f6046", "7c72a6df62704324b5abb2bb6d81453e", "1385631568564db5afe0e77016423c05", "74d2da7a2b624b21af1b4c8946b8ab21", "c427195ceb934510885d179b786dd7d8", "bbadc712055146a796786f377340e0eb", "545e6f2e748c4f4085f9608684b35482", "920eaadd6d75431eaecd1119996faed0", "565ae574d5f14dd090de6c9350ae465d", "b4d36110241a4e948127013ee4b3228c", "5cef4885408a495a970f0fdd3440bb7d", "69c3b96e88c54ddeb111c20ff9795d68", "6c23373b1ef9477086426c13bec22bd3", "f7f2b050a8d44dc9ac0e0aae63111aca", "47949a0cff2147bd942f1490fa79cd82", "a402ddcaf3754c4c8aaf9afb0cb36987", "432c53f85a6043b8aca2e36a379b7fc2", "3f764539d6fc4f67b7c1e42372f9b932", "7471767cb498487896de11f1f4fa3ad1", "d2580986fc214abfa4b4fcf8e4ed5fd2", "5ee2f6e4a4ac4de2854440a102251bfd", "a202db12cb8c4f6aa7cb5ef94876f1ca", "64a4d3ef169b40388b72c78f496ca102", "845108a47f5b4305b14f65d93b242638", "ab066fa759864fa7b415a8528fa66372", "93fff960cc22438f830253c3a009bc45", "e2e9a86edec74a1d8d8f6c2630b3697e", "49c299e91360455cb9b2afd3e72da85e", "480c5c3704434e18ac642281c9330c9e", "56861a83e9ab4eae90a31169b4160ca3", "2b1d3c85c9ee43feacbddedbe21579a1", "300edaa18082445684d9cfdbc8d28b44", "43c01118d6f74df4ba98a2822f58143b", "9e16da15118346efa8ace953608964ff", "0246d53d799f4c948d80065fd652022f", "ead5f0f310bb409ab32095f24f81f458", "f26fe72804fd409e8055753882029a0e", "f07d64ba1b6c4765828f4670aca060d7", "3efd4bd32ab34de8992e2cfbeff6e9e4", "fa19dc98f6dc4bc39aef713b77768afc", "ec344d3cdcca4744b542f2297cea9bdc", "d6fa3cd8e60545f4b41af0a040e11792", "8371d472094047a3878a060e47dd128b", "a13c4673847247199e37232fe5f22886", "f6071b90741d41e7a654e609030673dd", "2518f622d6de451eb38df7a26acdaee6", "bf0bfb2d3de44e929a6cb2713ee4d434"]} id="8HyL0YsC8vta" outputId="bbf536c1-1101-44bf-8a74-69a07bb21ccc"
from transformers import BertTokenizer, BertModel

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# + [markdown] id="lNO_QG48X-0C"
# Let's apply both tokenizers to some biomedical text and see how they compare. 
#
# I took the below sentence from the 2001 paper [Hydrogels for biomedical applications](http://yunus.hacettepe.edu.tr/~damlacetin/kmu407/index_dosyalar/Hoffman,%202012.pdf), which seems to be one of the most-cited papers in the field of biomedical applications (if I'm interpreting these [Google Scholar results](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=biomedical+applications&btnG=) correctly).

# + colab={"base_uri": "https://localhost:8080/"} id="2GAGQA0DQeIA" outputId="2e8f579d-d87d-4ee9-9077-0178fdd39055"
text = "Hydrogels are hydrophilic polymer networks which may absorb from " \
       "10–20% (an arbitrary lower limit) up to thousands of times their " \
       "dry weight in water."

bert_tokens = bert_tokenizer.tokenize(text)
scibert_tokens = scibert_tokenizer.tokenize(text)
bigbert_tokens = bigbert_tokenizer.tokenize(text)

len(bert_tokens), len(scibert_tokens), len(bigbert_tokens)

# + colab={"base_uri": "https://localhost:8080/"} id="CBVsoh_wR6hM" outputId="f210a5a2-65a7-4bc1-cc8d-dd8b1d38fe4d"
bert_tokens

# + colab={"base_uri": "https://localhost:8080/"} id="CUavmOrQR9rL" outputId="187f52e7-ec4e-4ba9-b00f-554729afedfd"
scibert_tokens

# + colab={"base_uri": "https://localhost:8080/"} id="QHhDeuzuSLOM" outputId="698525f0-7f08-46d4-c15e-125e79e4023e"
bigbert_tokens

# + colab={"base_uri": "https://localhost:8080/"} id="eht6-fWVYARe" outputId="1ce5686c-6a49-41e4-bc7e-048fd3908301"
text = "Hydrogels are hydrophilic polymer networks which may absorb from " \
       "10–20% (an arbitrary lower limit) up to thousands of times their " \
       "dry weight in water."

# Split the sentence into tokens, with both BERT and SciBERT.
bert_tokens = bert_tokenizer.tokenize(text)
scibert_tokens = scibert_tokenizer.tokenize(text)
bigbert_tokens = bigbert_tokenizer.tokenize(text)

# Pad out the scibert list to be the same length.
#while len(scibert_tokens) < len(bert_tokens):
#    scibert_tokens.append("")
#while len(bigbert_tokens) < len(bert_tokens):
#    scibert_tokens.append("")

# Label the columns.
print('{:<12} {:<12} {:<12}'.format("BERT", "SciBERT", "BigBERT"))
print('{:<12} {:<12} {:<12}'.format("----", "-------", "-------"))

# Display the tokens.
for tup in zip(bert_tokens, scibert_tokens, bigbert_tokens):
    print('{:<12} {:<12} {:<12}'.format(tup[0], tup[1], tup[2]))


# + [markdown] id="Y9KJ779ylBW8"
# SciBERT apparently has embeddings for the words 'hydrogels' and 'hydrophillic', whereas BERT had to break these down into three subwords each. (Remember that the '##' in a token is just a way to flag it as a subword that is not the first subword). Apparently BERT does have "polymer", though!
#
# I skimmed the paper and pulled out some other esoteric terms--check out the different numbers of tokens required by each model.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="OLfhdIHwdLnK" outputId="7aa8f6bc-9d6f-45ba-cdd5-1e9b415992a1"
# Use pandas just for table formatting.
import pandas as pd

# Some strange terms from the paper.
words = ['polymerization', 
         '2,2-azo-isobutyronitrile',
         'multifunctional crosslinkers',
         'carbon-capture-storage',
         'carbon sequestration',
         'co2'
         ]

# For each term...
for word in words:
    
    # Print it out
    print('\n\n', word, '\n')

    # Start a list of tokens for each model, with the first one being the model name.
    list_a = ["BERT:"]
    list_b = ["SciBERT:"]
    list_c = ["BigBERT:"]

    # Run both tokenizers.
    list_a.extend(bert_tokenizer.tokenize(word))
    list_b.extend(scibert_tokenizer.tokenize(word))
    list_c.extend(bigbert_tokenizer.tokenize(word))

    # Pad the lists to the same length.
    while len(list_a) < len(list_b):
        list_a.append("")
    while len(list_b) < len(list_a):
        list_b.append("")
    while len(list_a) < len(list_c):
        list_b.append("")

    # Wrap them in a DataFrame to display a pretty table.
    df = pd.DataFrame([list_a, list_b, list_c])
    
    display(df)


# + [markdown] id="kCl4ZLrzwkyq"
# The fact that SciBERT is able to represent all of these terms in fewer tokens seems like a good sign!

# + [markdown] id="7UB5FRPOhinf"
# ### Vocab Dump
#
# It can be pretty interesting just to dump the full vocabulary of a model into a text file and skim it to see what stands out.
#
# This cell will write out SciBERT's vocab to 'vocabulary.txt', which you can open in Colab by going to the 'Files' tab in the pane on the left and double clicking the .txt file.

# + id="NRf7NXJGgsnG"
with open("vocabulary.txt", 'w') as f:
    
    # For each token in SciBERT's vocabulary...
    for token in bigbert_tokenizer.vocab.keys():
        
        # Write it out, one per line.
        f.write(token + '\n')


# + [markdown] id="c8geTQ3Iidsv"
# You'll see that roughly the first 100 tokens are reserved, and then it looks like the rest of the vocabulary is sorted by frequency... The first actual tokens are:
#
# `t`, `a`, `##in`, `##he`, `##re`, `##on`, `the`, `s`, `##ti`
#
# > *Because the tokenizer breaks down "unknown" words into subtokens, it makes sense that some individual characters and subwords would be higher in frequency even than the most common words like "the".*

# + [markdown] id="zWU87CHqxgRt"
# ### Numbers and Symbols

# + [markdown] id="HFgt0WQ9Noch"
# There seem to be a lot of number-related tokens in SciBERT--you see them constantly as you scroll through the vocabulary. Here are some examples:

# + colab={"base_uri": "https://localhost:8080/"} id="pZY_Xd2iL3SS" outputId="2fef7b78-e706-49d0-c1e1-a314f7b98a65"
"##.2%)" in scibert_tokenizer.vocab

# + colab={"base_uri": "https://localhost:8080/"} id="FYR18oRbU9Yn" outputId="c47249e9-82da-4862-ae1b-42922420ec9b"
"##.2%)" in bigbert_tokenizer.vocab

# + colab={"base_uri": "https://localhost:8080/"} id="jaO-Eak5L_4P" outputId="33116f6d-c148-4e54-b813-531ec7008a1d"
"0.36" in scibert_tokenizer.vocab

# + colab={"base_uri": "https://localhost:8080/"} id="mdZCv9DzVCMX" outputId="da0a39a7-d45c-476d-fa8f-7d2a30f65717"
"0.36" in bigbert_tokenizer.vocab

# + [markdown] id="1aLXkM4MrAJ8"
# In the below loops, we'll tally up the number of tokens which include a digit, and show a random sample of these tokens. We'll do this for both SciBERT and BERT for comparison.
#

# + colab={"base_uri": "https://localhost:8080/"} id="jUFGHG_Kpmw6" outputId="97cf92b9-33e6-4b21-97c2-6a38c539bdd9"
import random

# ======== BERT ========
bert_examples = []

count = 0

# For each token in the vocab...
for token in bert_tokenizer.vocab:
    
    # If there's a digit in the token...
    # (But don't count those reserved tokens, e.g. "[unused59]")
    if any(i.isdigit() for i in token) and not ('unused' in token):
        # Count it.
        count += 1

        # Keep ~1% as examples to print.
        if random.randint(0, 100) == 1:
            bert_examples.append(token)

# Calculate the count as a percentage of the total vocab.
prcnt = float(count) / len(bert_tokenizer.vocab)

# Print the result.
print('In BERT:    {:>5,} tokens ({:.2%}) include a digit.'.format(count, prcnt))

# ======== SciBERT ========
scibert_examples = []
count = 0

# For each token in the vocab...
for token in scibert_tokenizer.vocab:

    # If there's a digit in the token...
    # (But don't count those reserved tokens, e.g. "[unused59]")
    if any(i.isdigit() for i in token) and not ('unused' in token):
        # Count it.
        count += 1

        # Keep ~1% as examples to print.
        if random.randint(0, 100) == 1:
            scibert_examples.append(token)
   

# Calculate the count as a percentage of the total vocab.
prcnt = float(count) / len(scibert_tokenizer.vocab)

# Print the result.
print('In SciBERT: {:>5,} tokens ({:.2%}) include a digit.'.format(count, prcnt))


# ======== BigBERT ========
bigbert_examples = []

count = 0

# For each token in the vocab...
for token in bigbert_tokenizer.vocab:
    
    # If there's a digit in the token...
    # (But don't count those reserved tokens, e.g. "[unused59]")
    if any(i.isdigit() for i in token) and not ('unused' in token):
        # Count it.
        count += 1

        # Keep ~1% as examples to print.
        if random.randint(0, 100) == 1:
            bigbert_examples.append(token)

# Calculate the count as a percentage of the total vocab.
prcnt = float(count) / len(bigbert_tokenizer.vocab)

# Print the result.
print('In BigBERT: {:>5,} tokens ({:.2%}) include a digit.'.format(count, prcnt))

print('')
print('Examples from BERT:', bert_examples)
print('Examples from SciBERT:', scibert_examples)
print('Examples from BigBERT:', bigbert_examples)

# + [markdown] id="-IY_eTkmscaK"
# So it looks like:
# - SciBERT has about 3x as many tokens with digits. 
# - BERT's tokens are whole integers, and many look like they could be dates. (In [another Notebook](https://colab.research.google.com/drive/1fCKIBJ6fgWQ-f6UKs7wDTpNTL9N-Cq9X#scrollTo=-M1biDEVYjaL), I showed that BERT contains 384 of the integers in the range 1600 - 2021).
# - SciBERT's number tokens are much more diverse. They are often subwords, and many include decimal places or  other symbols like `%` or `(`.

# + [markdown] id="j4uNOCxvNwcG"
#
# Random -- check out token 17740!:
#
# ⎝
#
# Looks like something is stuck to your monitor!  o_O

# + [markdown] id="NjzQ6dKzMW3A"
# ## 3.2. Comparing Embeddings
#

# + [markdown] id="d_A1kS9zHXp2"
# **Semantic Similarity on Scientific Text**
#
# To create a simple demonstration of SciBERT's value, Nick and I figured we could create a semantic similarity example where we show that SciBERT is better able to recognize similarities and differences within some scientific text than generic BERT. 
#
# We implemented this idea, but the examples we tried don't appear to show SciBERT as being better! 
#
# We thought our code and results are interesting to share all the same. 
#
# Also, while our simple example didn't succeed, note that the authors of SciBERT rigorously demonstrate its value over the original BERT by reporting results on a number of different NLP benchmarks that are focused specifically on scientific text. You can find the results in their paper [here](https://arxiv.org/abs/1903.10676). 
#
#
#

# + [markdown] id="7CSm_6ng1ufC"
# **Our Approach**
#
# In our semantic similarity task, we have three pieces of text--call them "query", "A", and "B", that are all on scientific topics. We pick these such that the query text is always more similar to A than to B. 
#
# Here's an example:
#
# * query: "Mitochondria (mitochondrion, singular) are membrane-bound cell organelles."
# * A: "These powerhouses of the cell produce adenosine triphosphate (ATP)."
# * B: "Ribosomes contain RNA and are responsible for synthesizing the proteins needed for many cellular functions."
#
# `query` and `A` are both about mitochondria, whereas `B` is about ribosomes. However, to recognize the similarity between `query` and `A`, you would need to know that mitochondria are responsible for producing ATP.  
#
# Our intuition was that SciBERT, being trained on biomedical text, would better distinguish the similarities than BERT. 
#
#
#
#

# + [markdown] id="CIBYucz_3yQQ"
# **Interpreting Cosine Similarities**
#
# When comparing two different models for semantic similarity, it's best to look at how well they *rank* the similarities, and not to compare the specific cosine similarity *values* across the two models.
#
# It's for this reason that we've structured our example as "is `query` more similar to `A` or to `B`?"
#

# + [markdown] id="PJZL5KflODld"
# **Embedding Functions**
#
# In order to try out different examples, we've defined a `get_embedding` function below. It takes the average of the embeddings from the second-to-last layer of the model to use as a sentence embedding.
#
# `get_embedding` also supports calculating an embedding for a specific word or sequence of words within the sentence. 
#
# To locate the indeces of the tokens for these words, we've also defined the `get_word_indeces` helper function below. 
#
# To calculate the word embedding, we again take the average of its token embeddings from the second-to-last layer of the model.
#

# + [markdown] id="NMLNJuNCiHE6"
# #### get_word_indeces
#

# + id="BMORSn8rmtJf"
import numpy as np

def get_word_indeces(tokenizer, text, word):
    '''
    Determines the index or indeces of the tokens corresponding to `word`
    within `text`. `word` can consist of multiple words, e.g., "cell biology".
    
    Determining the indeces is tricky because words can be broken into multiple
    tokens. I've solved this with a rather roundabout approach--I replace `word`
    with the correct number of `[MASK]` tokens, and then find these in the 
    tokenized result. 
    '''
    # Tokenize the 'word'--it may be broken into multiple tokens or subwords.
    word_tokens = tokenizer.tokenize(word)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]']*len(word_tokens))

    # Replace the word with mask tokens.
    text_masked = text.replace(word, masks_str)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(text_masked)

    # Use numpy's `where` function to find all indeces of the [MASK] token.
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces



# + id="_lhBdgiIdRcm"
text = "capture co2"
word = "co2"

# + colab={"base_uri": "https://localhost:8080/"} id="tssrlCG6daq_" outputId="aae315c2-b75a-4ede-bb57-8c9c3997733c"
bigbert_tokenizer.tokenize(text)

# + colab={"base_uri": "https://localhost:8080/"} id="RBCkFxB2YyB3" outputId="e169af6c-c069-459c-e3cf-0b841ff4222e"
word_tokens = bigbert_tokenizer.tokenize(word)
word_tokens

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="e0WkK0BZdAGW" outputId="1ea35437-06cc-4097-bb2f-fb4e7f30af3a"
masks_str = ' '.join(['[MASK]']*len(word_tokens))
masks_str

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="SDG_3xrVdQkX" outputId="feb776f4-a7ca-462e-a6f3-af26b8f44855"
text_masked = text.replace(word, masks_str)
text_masked

# + colab={"base_uri": "https://localhost:8080/"} id="vibUX8UQZGiI" outputId="654e28e9-bdb3-4f4a-fb53-5cb9607d52e0"
bigbert_tokenizer.encode(text.replace("co2", ' '.join(['[MASK]']*2)))

# + colab={"base_uri": "https://localhost:8080/"} id="nm1BNcMkdtbQ" outputId="a7319586-e778-4cd9-9ec5-2a4c7039d904"
bigbert_tokenizer.encode("capture")

# + colab={"base_uri": "https://localhost:8080/"} id="UuZPuV42d3dn" outputId="eb18b880-a6ce-4baf-9d6f-5f4ef306da9e"
bigbert_tokenizer.encode("co2")

# + colab={"base_uri": "https://localhost:8080/"} id="YA_m8CtBaXSc" outputId="61883079-ef39-4b97-f785-b26a70aae25a"
bigbert_tokenizer.mask_token_id

# + colab={"base_uri": "https://localhost:8080/"} id="MlQMT-ewYh4X" outputId="e86c8748-53a2-4fe5-eb3f-5e70017ddc93"
get_word_indeces(bigbert_tokenizer, text, word)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["40395acb6d2e45acb597c6ba54fcc90e", "3a3695dc1e5b4c009a7651d1b6bebf41", "93cb97b947fa4613b40e21b2ed5ffee8", "4e55795a139244fcb4f91994c7cdc6b4", "d2872aa16a14460cb32ec0bbaa915543", "3430e5694dc747839bbd95ccb6c3ad5c", "03e209c82555453d9716023f344eb850", "067aa84a09f8421eb8d7fe2c75742337", "033c2db26a9f41af867d7d6452b21b71", "4c1273a4c1604750aedabfa2df7dbaab", "6b4202be39814b0f96d8e201cd9f302f", "57bbcfd989364e7186515d7a40fc9a01", "c584bf441d0a48dc9a1e2f19744e7746", "77ef3f66ce4f4af08a72ec5166e93c0f", "908a2db2ee3948428773133f89eed921", "b8c99d821a5846efa6bed7b35eeb6bac", "96d0d26d76f343499afae34e3ab28c2c", "4031f4738fc848c6b6a1210427c00956", "5e580aef7d3f465db41f1aad931644ba", "3d69b00764734a719aa03f77dc06ac78", "bec57984ebb44dc6ad4ea941212d4f9a", "f77410f1b96e495aaea1008c3721d2be", "accaac95ff4b4f56a93640601919c72f", "48c4b8ecc69e49be8e7655526b8f04dd", "20debbfa065940bbb5f5b0e51ac2a655", "364f5734ee1043968700bf64f7b7ac1a", "e891117f4b29415b879a7b22f201dff7", "82e18bfe9daf4f18adcf78c803f1ca3a", "7ff832619da943bbbc4672fc0777808f", "b1ecbe981bc542e78dafe56da3fd96e9", "bdd6c7fffe4249ffa59c5db80ef539fb", "3e8d77778c7242089d3c2f7a3a5c3ab4", "aeede1af6a2b4cf99fcf57a5ce795120"]} id="w_dkzN94eKCm" outputId="5cba26e4-c9be-4bf0-e891-c533d891ed73"
from transformers import AutoTokenizer, AutoModel
import torch

# Retrieve BigBERT.
bigbert_model = AutoModel.from_pretrained("anferico/bert-for-patents",
                                  output_hidden_states=True)
bigbert_tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")

bigbert_model.eval()

# + id="GKYaehJHfTpA"
text = "capture co2"
word = "co2"

# + colab={"base_uri": "https://localhost:8080/"} id="hBAUUuP5fTpA" outputId="26082222-f5dc-4ddc-9053-eb780685629d"
bigbert_tokenizer.tokenize(text)

# + colab={"base_uri": "https://localhost:8080/"} id="mdehjNz-fTpB" outputId="cfc7ba12-ad45-4d51-9095-0cae81c0244a"
word_tokens = bigbert_tokenizer.tokenize(word)
word_tokens

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="GvtL60r-fTpB" outputId="dc6c9d0c-ca35-4b41-cd53-05826b8d8a1a"
masks_str = ' '.join(['[MASK]']*len(word_tokens))
masks_str

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="VIi7ony-fTpB" outputId="cbfee4ff-4d70-4ccd-b5ba-bb72d6f8d5e2"
text_masked = text.replace(word, masks_str)
text_masked

# + colab={"base_uri": "https://localhost:8080/"} id="y2bbUMYOfTpB" outputId="d03a5dbb-d5a3-4f42-8901-e76570cdfb65"
bigbert_tokenizer.encode(text.replace("co2", ' '.join(['[MASK]']*2)))

# + colab={"base_uri": "https://localhost:8080/"} id="_wgeMoB7fTpB" outputId="0b023890-a8c7-430d-cd67-a76c83550f17"
bigbert_tokenizer.encode("capture")

# + colab={"base_uri": "https://localhost:8080/"} id="j8WCHgJOfTpB" outputId="dc4052d4-7e24-447f-e1cc-eff7c7d4eba1"
bigbert_tokenizer.encode("co2")

# + colab={"base_uri": "https://localhost:8080/"} id="AYND_df2fTpB" outputId="84154a58-5085-46ff-ba39-493425e717db"
bigbert_tokenizer.mask_token_id

# + colab={"base_uri": "https://localhost:8080/"} id="kLKKCt5GkX2w" outputId="0d7c2868-ee61-4020-e088-1398b9e038f8"
bigbert_tokenizer.cls_token_id

# + colab={"base_uri": "https://localhost:8080/"} id="OBoI0tUsfTpB" outputId="d9e3b92c-26ff-4aaf-9734-54f74903d779"
get_word_indeces(bigbert_tokenizer, text, word)


# + [markdown] id="_vclJjNvwlmx"
# #### get_embedding

# + id="giJMGiGZhLLa"
def get_embedding(b_model, b_tokenizer, text, word=''):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    '''

    # If a word is provided, figure out which tokens correspond to it.
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors.
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    input_ids = encoded_dict['input_ids']
    
    b_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = b_model(input_ids)
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

        outputs = b_model(input_ids)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    if not word == '':
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()
    
        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding



# + colab={"base_uri": "https://localhost:8080/"} id="VrvWakGMh2DN" outputId="7d63017b-1b51-4290-b411-5115352ed6b3"
encoded_dict = bigbert_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )
encoded_dict

# + colab={"base_uri": "https://localhost:8080/"} id="gIlkgd30hf2j" outputId="70c791ae-7b82-4ddd-fa3e-f7fbcaf4b86f"
get_embedding(bigbert_model, bigbert_tokenizer, text, word)

# + colab={"base_uri": "https://localhost:8080/"} id="MGOHjnxfiJCN" outputId="50cb1ac8-38e7-499c-b092-a747ae456a07"
encoded_dict = bert_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )
encoded_dict

# + [markdown] id="6FyjdEFRwUty"
# Retrieve the models and tokenizers for both BERT and SciBERT

# + colab={"base_uri": "https://localhost:8080/"} id="omt9VRwYoJSB" outputId="aa9bdc15-81fd-4a76-926a-1980ab246db6"
from transformers import BertModel, BertTokenizer

# Retrieve SciBERT.
scibert_model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased",
                                  output_hidden_states=True)
scibert_tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

scibert_model.eval()

# Retrieve generic BERT.
bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True) 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model.eval()

# + [markdown] id="GP_Uv1i8w9FV"
# Test out the function.

# + colab={"base_uri": "https://localhost:8080/"} id="PGoJ15Mmik_F" outputId="c3b75437-4cee-486f-ff7f-06204ec69925"
text = "hydrogels are hydrophilic polymer networks which may absorb from 10–20% (an arbitrary lower limit) up to thousands of times their dry weight in water."
word = 'hydrogels'

# Get the embedding for the sentence, as well as an embedding for 'hydrogels'.
(sen_emb, word_emb) = get_embedding(scibert_model, scibert_tokenizer, text, word)

print('Embedding sizes:')
print(sen_emb.shape)
print(word_emb.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="V3xA0CispwaK" outputId="d426f3dd-7616-4721-e04d-acc8fdc0904b"
text = "hydrogels are hydrophilic polymer networks which may absorb from 10–20% (an arbitrary lower limit) up to thousands of times their dry weight in water."
word = 'hydrogels'

# Get the embedding for the sentence, as well as an embedding for 'hydrogels'.
(sen_emb, word_emb) = get_embedding(bigbert_model, scibert_tokenizer, text, word)

print('Embedding sizes:')
print(sen_emb.shape)
print(word_emb.shape)

# + [markdown] id="6MensSbiw4nn"
# Here's the code for calculating cosine similarity. We'll test it by comparing the word embedding with the sentence embedding--not a very interesting comparison, but a good sanity check.

# + colab={"base_uri": "https://localhost:8080/"} id="nPNIJ8vRw4-K" outputId="f15f8d5f-5ece-4d48-8ef7-281e6a31853b"
from scipy.spatial.distance import cosine

# Calculate the cosine similarity of the two embeddings.
sim = 1 - cosine(sen_emb, word_emb)

print('Cosine similarity: {:.2}'.format(sim))

# + [markdown] id="mYGiIc6DPEJL"
# #### Sentence Comparison Examples

# + [markdown] id="yyF4JRmjtRlO"
# In this example, `query` and `A` are about biomedical "hydrogels", and `B` is from astrophysics.
#
# Both models make the correct distinction, but generic BERT seems to be better...

# + colab={"base_uri": "https://localhost:8080/"} id="SoV1kYclsvUf" outputId="b592c68f-371e-45fa-a8ea-bcc9361c0711"
# Three sentences; query is more similar to A than B.
text_query = "Chemical hydrogels are commonly prepared in two different ways: ‘three-dimensional polymerization’ (Fig. 1), in which a hydrophilic monomer is polymerized in the presence of a polyfunctional cross-linking agent, or by direct cross-linking of water-soluble polymers (Fig. 2)."
text_A = "Hydrogels can be obtained by radiation technique in a few ways, including irradiation of solid polymer, monomer (in bulk or in solution), or aqueous solution of polymer."
text_B = "The observed cosmic shear auto-power spectrum receives an additional contribution due to shape noise from intrinsic galaxy ellipticities."

# Get embeddings for each.
emb_query = get_embedding(scibert_model, scibert_tokenizer, text_query)
emb_A = get_embedding(scibert_model, scibert_tokenizer, text_A)
emb_B = get_embedding(scibert_model, scibert_tokenizer, text_B)

# Compare query to A and B with cosine similarity.
sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print("'query' should be more similar to 'A' than to 'B'...\n")

print('SciBERT:')
print('  sim(query, A): {:.2}'.format(sim_query_A))
print('  sim(query, B): {:.2}'.format(sim_query_B))

# Repeat with BERT.
emb_query = get_embedding(bert_model, bert_tokenizer, text_query)
emb_A = get_embedding(bert_model, bert_tokenizer, text_A)
emb_B = get_embedding(bert_model, bert_tokenizer, text_B)

# Compare query to A and B with cosine similarity.
sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print('')
print('BERT:')
print('  sim(query, A): {:.2}'.format(sim_query_A))
print('  sim(query, B): {:.2}'.format(sim_query_B))


# + [markdown] id="WtnsgeBpuOMx"
# In this example, `query` and `A` are both about mitochondria, while `B` is about ribosomes. 
#
# Neither model seems to recognize the distinction!
#
#

# + colab={"base_uri": "https://localhost:8080/", "height": 181} id="VbD4lvuBO4B9" outputId="df558be7-1ec2-4438-8c0e-466483ea81cf"
# Three sentences; query is more similar to A than B.
text_query = "Mitochondria (mitochondrion, singular) are membrane-bound cell organelles."
text_A = "These powerhouses of the cell produce adenosine triphosphate (ATP)." 
text_B = "Ribosomes contain RNA and are responsible for synthesizing the proteins needed for many cellular functions."
#text_B = "Molecular biology deals with the structure and function of the macromolecules (e.g. proteins and nucleic acids) essential to life." 

# Get embeddings for each.
emb_query = get_embedding(scibert_model, scibert_tokenizer, text_query)
emb_A = get_embedding(scibert_model, scibert_tokenizer, text_A)
emb_B = get_embedding(scibert_model, scibert_tokenizer, text_B)

# Compare query to A and B with cosine similarity.
sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print("'query' should be more similar to 'A' than to 'B'...\n")

print('SciBERT:')
print('  sim(query, A): {:.2}'.format(sim_query_A))
print('  sim(query, B): {:.2}'.format(sim_query_B))

# Repeat with BERT.
emb_query = get_embedding(bert_model, bert_tokenizer, text_query)
emb_A = get_embedding(bert_model, bert_tokenizer, text_A)
emb_B = get_embedding(bert_model, bert_tokenizer, text_B)

# Compare query to A and B with cosine similarity.
sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print('')
print('BERT:')
print('  sim(query, A): {:.2}'.format(sim_query_A))
print('  sim(query, B): {:.2}'.format(sim_query_B))


# + [markdown] id="WOmJjM5KyxsZ"
# #### Word Comparison Examples
#

# + [markdown] id="TwVtyfuR7xoF"
#
# We also payed with comparing words that have both scientific and non-scientific meaning. For example, the word "cell" can refer to biological cells, but it can also refer (perhaps more commonly) to prison cells, cells in Colab notebooks, cellphones, etc.
#
# In this example we'll use the word "cell" in a sentence with two other words that evoke its scientific and non-scientific usage: "animal" and "prison." 
#
# > "The man in prison watched the animal from his cell."
#
# Both BERT and SciBERT output "contextualized" embeddings, meaning that the representation of each word in a sentence will change depending on the words that occur around it.
#
# In our example sentence, it's clear from the context that "cell" refers to *prison cell*, but we theorized that SciBERT would be more biased towards the biological interpretation of the word. The result below seems to confirm this.

# + colab={"base_uri": "https://localhost:8080/", "height": 181} id="BpUdEK5k8xx_" outputId="d29235f5-bd16-42b1-bf7d-45ecb5b78ea3"
text = "The man in prison watched the animal from his cell."

print('"' + text + '"\n')

# ======== SciBERT ========

# Get contextualized embeddings for "prison", "animal", and "cell"
(emb_sen, emb_cell) = get_embedding(scibert_model, scibert_tokenizer, text, word="cell")
(emb_sen, emb_prison) = get_embedding(scibert_model, scibert_tokenizer, text, word="prison")
(emb_sen, emb_animal) = get_embedding(scibert_model, scibert_tokenizer, text, word="animal")

# Compare the embeddings
print('SciBERT:')
print('  sim(cell, animal): {:.2}'.format((1 - cosine(emb_cell, emb_animal))))
print('  sim(cell, prison): {:.2}'.format(1 - cosine(emb_cell, emb_prison)))

print('')

# ======== BERT ========

# Get contextualized embeddings for "prison", "animal", and "cell"
(emb_sen, emb_cell) = get_embedding(bert_model, bert_tokenizer, text, word="cell")
(emb_sen, emb_prison) = get_embedding(bert_model, bert_tokenizer, text, word="prison")
(emb_sen, emb_animal) = get_embedding(bert_model, bert_tokenizer, text, word="animal")

# Compare the embeddings
print('BERT:')
print('  sim(cell, animal): {:.2}'.format((1 - cosine(emb_cell, emb_animal))))
print('  sim(cell, prison): {:.2}'.format(1 - cosine(emb_cell, emb_prison)))


# + [markdown] id="FYq0v6O3-tVG"
# Let us know if you find some more interesting examples to try!

# + [markdown] id="-B36npqKZkph"
# # Appendix: BioBERT vs. SciBERT

# + [markdown] id="J51vlJa9Zm_c"
# I don't have much insight into the merits of BioBERT versus SciBERT, but I thought I would at least share what I do know.
#
# **Publish Dates & Authors**
#
# * *BioBERT*
#     * First submitted to arXiv: `Jan 25th, 2019`
#         * [link](https://arxiv.org/abs/1901.08746)
#     * First Author: Jinhyuk Lee
#     * Organization: Korea University, Clova AI (also Korean)
#
# * *SciBERT*
#    * First submitted to arXiv: `Mar 26, 2019`
#        * [arXiv](https://arxiv.org/abs/1903.10676), [pdf](https://arxiv.org/pdf/1903.10676.pdf)
#     * First Author: Iz Beltagy
#     * Organization: Allen AI
#
# **Differences**
#
# * BioBERT used the same tokens as the original BERT, rather than choosing a new vocabulary of tokens based on their corpus. Their justification was "to maintain compatibility", which I don't entirely understand.
# * SciBERT learned a new vocabulary of tokens, but they also found that this detail is less important--it's training on the specialized corpus that really makes the difference.
# * SciBERT is more recent, and outperforms BioBERT on many, but not all, scientific NLP benchmarks.
# * The difference in naming seems unfortunate--SciBERT is also trained primarily on biomedical research papers, but the name "BioBERT" was already taken, so....
#
# **huggingface transformers**
#
# * Allen AI published their SciBERT models for the transformers library, and you can see their popularity:
#     * [SciBERT uncased](https://huggingface.co/allenai/scibert_scivocab_uncased): ~16.7K downloads (from 5/22/20 - 6/22/20)
#         * `allenai/scibert_scivocab_uncased`
#     * [SciBERT cased](https://huggingface.co/allenai/scibert_scivocab_cased ): ~3.8k downloads (from 5/22/20 - 6/22/20)
#         * `allenai/scibert_scivocab_cased`
# * The BioBERT team has published their models, but not for the `transformers` library, as far as I can tell. 
#     * The most popular BioBERT model in the huggingface community appears to be [this one](https://huggingface.co/monologg/biobert_v1.1_pubmed): `monologg/biobert_v1.1_pubmed`, with ~8.6K downloads (from 5/22/20 - 6/22/20)
#        * You could also download BioBERT's pre-trained weights yourself from https://github.com/naver/biobert-pretrained, but I'm not sure what it would take to pull these into the `transformers` library exactly. 
#
#
