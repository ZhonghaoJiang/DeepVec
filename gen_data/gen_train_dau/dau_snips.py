import pandas as pd
from nltk.corpus import wordnet
from sklearn.feature_extraction import text
stop_words = set(text.ENGLISH_STOP_WORDS)
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import random
import nlpaug.flow as nafc
from nlpaug.util import Action
import csv
import os

# from nlpaug.util.file.download import DownloadUtil
# DownloadUtil.download_word2vec(dest_dir='./nlp_model/')  # Download word2vec model
# DownloadUtil.download_glove(model_name='glove.6B', dest_dir='./nlp_model/')  # Download GloVe model
# DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='./nlp_model/')  # Download fasttext model
model_dir = "../nlp_model/"
aug_synonym = naw.SynonymAug(aug_src='wordnet')
aug_replace = naw.WordEmbsAug(
        model_type='word2vec', model_path=model_dir + 'GoogleNews-vectors-negative300.bin',
        action="substitute")
back_translation_aug = naw.BackTranslationAug(
    from_model_name='transformer.wmt19.en-de',
    to_model_name='transformer.wmt19.de-en',
    device='cpu'
)
aug_insert = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")


def replace_synonym(text):
    augmented_text = aug_synonym.augment(text)
    return augmented_text


def replace_word2vec(text):
    augmented_text = aug_replace.augment(text)
    return augmented_text


def back_translate(text):
    augmented_text = back_translation_aug.augment(text)
    return augmented_text


def bert_insert(text):
    augmented_text = aug_insert.augment(text)
    return augmented_text


def random_aug1(sentence):
    augmentation = ['synonym_replace', 'word2vec_replace', 'back_translate', 'bert_insert']
    choose_aug = random.choice(augmentation)
    return aug_sentence(sentence, choose_aug)


def aug_sentence(sentence, choose_aug):
    aug = ""
    if choose_aug == "synonym_replace":
        aug = replace_synonym(sentence)
    elif choose_aug == "word2vec_replace":
        aug = replace_word2vec(sentence)
    elif choose_aug == "back_translate":
        aug = back_translate(sentence)
    elif choose_aug == "bert_insert":
        aug = bert_insert(sentence)
    return aug


if __name__ == '__main__':

    os.makedirs("./dau/snips_harder", exist_ok=True)
    data = pd.read_csv("../new_intent.csv")
    train_data = data[500:9990]
    test_data = data[9990:]

    selected_data_train = train_data.reset_index(drop=True)
    selected_data_test = test_data.reset_index(drop=True)
    print(len(train_data))
    # print(selected_data)

    train_aug = pd.DataFrame(columns=('text', 'intent'))
    test_aug = pd.DataFrame(columns=('text', 'intent'))
    idx = 0

    aug_text_li = []
    aug_intent_li = []
    ori_text_li = []
    ori_intent_li = []

    for i in range(len(selected_data_train)):
        ori_text = selected_data_train.text[i]
        ori_intent = selected_data_train.intent[i]

        aug_text_li.append(random_aug1(ori_text))
        aug_intent_li.append(ori_intent)

        ori_text_li.append(ori_text)
        ori_intent_li.append(ori_intent)

        print("processed:", i)

    for text, intent in zip(ori_text_li, ori_intent_li):
        aug = {'text': text, 'intent': intent}
        train_aug.loc[idx] = aug
        idx = idx + 1

    for text, intent in zip(aug_text_li, aug_intent_li):
        aug = {'text': text, 'intent': intent}
        train_aug.loc[idx] = aug
        idx = idx + 1


    train_aug.to_csv("./dau/snips_harder/snips_train_aug.csv")

    aug_text_li = []
    aug_intent_li = []
    ori_text_li = []
    ori_intent_li = []
    idx = 0

    for i in range(len(selected_data_test)):
        ori_text = selected_data_test.text[i]
        ori_intent = selected_data_test.intent[i]

        aug_text_li.append(random_aug1(ori_text))
        aug_intent_li.append(ori_intent)

        ori_text_li.append(ori_text)
        ori_intent_li.append(ori_intent)

        print("processed:", i)

    for text, intent in zip(aug_text_li, aug_intent_li):
        aug = {'text': text, 'intent': intent}
        test_aug.loc[idx] = aug
        idx = idx + 1

    test_aug.to_csv("./dau/snips_harder/snips_test_aug.csv")

 
