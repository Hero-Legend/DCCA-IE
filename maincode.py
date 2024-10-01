import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import Linear as DenseLinear
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dgl
import numpy as np
import torch as th
from dgl.nn import GATConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

#############################################定义 BERT 模型和 tokenizer##############################################

#导入Biobert
model_path = './model_path/biobert'                     #这个要用相对路径，不要用绝对路径
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path)


# #导入bert
# model_path_1 = './model_path/bert_pretrain'                     #这个要用相对路径，不要用绝对路径
# bert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
# bert_model = AutoModel.from_pretrained(model_path_1)



####################################################################################################################

#############################################读取数据################################################################

df_train = pd.read_csv('./data/ddi2013ms/train.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)

####################################################################################################################

#######################################################定义模型参数##################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

num_label=5

# 定义模型参数
max_length = 300
batch_size = 8


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def construct_txt_intra_matrix(self, word_num):
        """构建文本模态内的矩阵"""
        mat = np.zeros((max_length, max_length), dtype=np.float32)
        mat[:word_num, :word_num] = 1.0
        return mat

    def __getitem__(self, idx):
        sentence = str(self.data['sentence'][idx])
        label_str = self.data['label'][idx]
        label = label_map[label_str]

        encoding = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
 
        # 使用 attention_mask 来确定有效的 token 数量
        word_num = encoding['attention_mask'].sum().item()
        txt_intra_matrix = self.construct_txt_intra_matrix(word_num)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'txt_intra_matrix': torch.tensor(txt_intra_matrix, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer=tokenizer,

        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 设置 drop_last=True 来丢弃最后一个不满足批次大小的批次,因为我们在LSTM和GCN维度转换时，出现了维度不匹配问题，找了很久原因，发现是在最后batch时，数据只有4条，导致维度出错
    )



# 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, biobert_tokenizer, max_length, batch_size)

# for data in train_data_loader:
#     print(data)
#     break  # 这将打印第一批数据并中断循环。


dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, max_length, batch_size)

# for batch in test_data_loader:
#     print(batch)
#     break  # 这将打印第一批数据并中断循环。


#BioBERT-CNN通道
class BioBERT_CNN(nn.Module):
    def __init__(self, hidden_dim=256):
        super(BioBERT_CNN, self).__init__()
        self.Biobert = biobert_model
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.Biobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, 768)
        sequence_output_transposed = sequence_output.transpose(1, 2)  # (batch_size, 768, seq_length)
        cnn_features = F.relu(self.conv1d(sequence_output_transposed)).transpose(1, 2)  # (batch_size, seq_length, hidden_dim)
        return cnn_features




#图通道
class BertBiLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=256, freeze_bert=True):
        super(BertBiLSTM, self).__init__()
        self.bert = biobert_model
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 256)  # 添加线性层

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        lstm_output, _ = self.lstm(sequence_output)
        output_features = self.linear(lstm_output)  # 应用线性层

        return output_features


# 将节点特征和关系矩阵转换为 DGL 图对象
def create_dgl_graph(fea, mat, device):

    # 获取节点数量和特征维度
    num_nodes = fea.shape[1]
    feat_dim = fea.shape[2]

    # 创建一个 DGL 图对象
    g = dgl.graph(([], []), num_nodes=num_nodes)    


    # 将图和节点特征移动到相同的设备上
    g = g.to(device)
    fea = fea.to(device)

    # 在图中添加自环
    g = dgl.add_self_loop(g)
    

    # 将节点特征添加到图中
    g.ndata['feat'] = fea.transpose(0, 1) # 调整特征的维度顺序并添加到图中

    return g


class GCNRelationModel(nn.Module):
    def __init__(self, d_model=512, d_hidden=256, dropout=0.2):
        super().__init__()
        
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden

        #####################BERT_Bi-LSTM作为嵌入层，它的输出作为特征
        self.BertBiLSTM = BertBiLSTM(hidden_dim=256, num_layers=1, num_classes=256, freeze_bert=True)


        # gat layer
        self.GATConv = GATConv(in_feats=256, out_feats=256, num_heads=3)

        # output mlp layers
        self.MLP = nn.Linear(256*3, 256)

    def forward(self, input_ids, attention_mask, labels, mat):
	    
        # 获取当前模型所在的设备
        device = input_ids.device

	    # 节点特征
        # print(input_ids.shape) # [8,300]
        fea = self.BertBiLSTM(input_ids, attention_mask)
        # print("fea:", fea.shape) 

    
        # 创建 DGL 图对象
        g = create_dgl_graph(fea, mat, device)
        


	   # 图神经网络
        outputs = self.GATConv(g, g.ndata['feat'])
        # print("GATConv outputs:", outputs.shape)     


        # 从 [300, 8, 3, 256] 重塑为 [batch_size, seq_length, input_size]
        outputs = outputs.permute(1, 0, 2, 3).contiguous()
        # print("outputs:", outputs.shape)      


       # 将 GATConv 的输出传递给 MLP
        outputs = outputs.view(8, 300, -1)  
        # print("outputs.view:", outputs.shape) 
        


        # Pass GATConv output through MLP
        gcn_features = self.MLP(outputs)

        # print("gcn_features_MLP:",gcn_features.shape)     #输出形状为 [8,300,256]
        return gcn_features


#交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        super(CrossAttention, self).__init__()


        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

    def forward(self, cnn_features, gcn_features):

        # print("GCN Features Shape:", gcn_features.shape)  # 添加此行以打印 gcn_features 的形状 
        # print("CNN Features Shape:", cnn_features.shape)  # [8,300,256]


        # 使用CNN特征作为查询（query），GCN特征作为键（key）和值（value）
        attn_output, attn_weights = self.attention(query=cnn_features, key=gcn_features, value=gcn_features)

        # # 打印注意力输出的形状
        # print("Attention Output Shape:", attn_output.shape)     

        return attn_output


#分类器
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features.mean(dim=1))  # 对特征进行平均池化，然后进行分类
        return logits


#整体模型
class BioMedRelationExtractor(nn.Module):
    def __init__(self):
        super(BioMedRelationExtractor, self).__init__()
        #self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bio_bert_cnn = BioBERT_CNN()
        self.GCNRelationModel = GCNRelationModel()
        self.cross_attention = CrossAttention()
        self.classifier = Classifier()

    def forward(self, input_ids, attention_mask, labels, mat):

        # 通过BioBERT-CNN通道获取特征
        cnn_features = self.bio_bert_cnn(input_ids, attention_mask)
        # print(cnn_features.shape)      #[8,300,256]

        # 通过BiLSTM-GCN通道获取特征
        gcn_features = self.GCNRelationModel(input_ids, attention_mask, labels, mat)
        # print(cnn_features.shape)      #[8,300,256]

        # 交叉注意力机制
        attn_output = self.cross_attention(cnn_features, gcn_features)   
        # print("attn_output:",attn_output.shape)             #输出形状为[8,300,256]

        # 分类器进行分类
        logits = self.classifier(attn_output)
        return logits


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    true_labels = []
    pred_labels = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        mat = batch['txt_intra_matrix'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels, mat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds
    
    # 计算混淆矩阵和 F1 值
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    return train_loss, train_acc, conf_matrix, f1


# 测试代码
def test_model(model, test_data_loader, criterion, device):
    # model.eval()
    # running_loss = 0.0
    # correct_preds = 0
    # total_preds = 0

    # with torch.no_grad():
    #     for batch in test_data_loader:
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         mat = batch['txt_intra_matrix'].to(device)

    #         outputs = model(input_ids, attention_mask, labels, mat)
    #         loss = criterion(outputs, labels)

    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs, 1)
    #         total_preds += labels.size(0)
    #         correct_preds += (predicted == labels).sum().item()

    # test_loss = running_loss / len(test_data_loader)
    # test_acc = correct_preds / total_preds
    # return test_loss, test_acc

    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mat = batch['txt_intra_matrix'].to(device)

            outputs = model(input_ids, attention_mask, labels, mat)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # 计算混淆矩阵、准确率、精确率、召回率和 F1 值
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    return conf_matrix, accuracy, precision, recall, f1


#模型实例化
model = BioMedRelationExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 25

for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    train_loss, train_acc, conf_matrix, f1 = train_model(model, train_data_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)



# 在测试集上调用 test_model 函数并打印输出信息
test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1 = test_model(model, test_data_loader, criterion, device)
print("Test Results:")
print("Confusion Matrix:")
print(test_conf_matrix)
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)
