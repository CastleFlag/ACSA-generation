import pandas as pd
import torch
from torch import tensor
import torch.nn.functional as F
from model import MyTokenizer
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import id4_to_entity, id7_to_entity, entity_to_id4, entity_to_id7

def create_dataloader(path, tokenizer, args):
    json_data = jsonlload(path)
    entity_dataset, polarity_dataset = get_dataset(json_data, tokenizer, args.max_len, args.num_labels)
    return DataLoader(entity_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0), DataLoader(polarity_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

class MyDataset(Dataset):
    def __init__(self, id_list, attention_mask_list, token_label_list):
        self.ids = id_list
        self.masks = attention_mask_list
        self.labels = token_label_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        return id, mask, label

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}
entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']

def tokenize_and_align_labels(tokenizer, form, annotations, max_len, num_labels):
    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
    # Entity has annotation
    if annotations:
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            if polarity == '------------':
                continue

            entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
            entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            if num_labels == 4:
                entity_property_data_dict['label'].append(entity_to_id4(entity_property))
            elif num_labels == 7:
                entity_property_data_dict['label'].append(entity_to_id7(entity_property))

            polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
            polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            polarity_data_dict['label'].append(polarity_name_to_id[polarity])
    else:        
        entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
        entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
        entity_property_data_dict['label'].append(0)
        polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
        polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
        polarity_data_dict['label'].append(0)
    return entity_property_data_dict, polarity_data_dict

def get_dataset(raw_data, tokenizer, max_len, num_labels):
    entity_input_ids_list = []
    entity_attention_mask_list = []
    entity_token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []
    
    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict  = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len, num_labels)

        entity_input_ids_list.extend(entity_property_data_dict['input_ids'])
        entity_attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        entity_token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])
    # print('entity_property_data_count: ', entity_property_count)
    # print('polarity_data_count: ', polarity_count)

    return MyDataset(tensor(entity_input_ids_list), tensor(entity_attention_mask_list), tensor(entity_token_labels_list)), MyDataset(tensor(polarity_input_ids_list), tensor(polarity_attention_mask_list), tensor(polarity_token_labels_list))