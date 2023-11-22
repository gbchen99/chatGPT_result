# 任务器类:封装方法，可训练模型，测试模型
import math
import pickle
import json
import tqdm

import jieba
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
from datasets import load_metric
from transformers import TrainingArguments, DataCollatorWithPadding, AutoConfig, IntervalStrategy, EarlyStoppingCallback
from typing import Union, Optional

from dataset.dataset_main import CAILDataset
from models.net_main_articles_att_correct import CAILNet as Net
from models.net_main_articles_att_correct import CAILNet as NetCNN
from models.net_main_articles_att_correct import CAILNet as NetPlus
from utils.dev_submit import comput_dev, check_filename_available
from utils.my_trainer import MyTrainer
from utils.pub_utils import save_submition, accuracy_compute_objective
# from process.bm25 import process_input

warnings.filterwarnings("ignore")


class BM25Param(object):
    def __init__(self, f, df, idf, length, avg_length, docs_list, line_length_list, k1=1.5, k2=1.0, b=0.75):
        """

        :param f:
        :param df:
        :param idf:
        :param length:
        :param avg_length:
        :param docs_list:
        :param line_length_list:
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param b:
        """
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"


class BM25(object):
    #     _param_pkl = "data/param.pkl"
    #     _docs_path = "data/data.txt"
    # 此处的result是拼接了所有法条，司法解释等
    root_dir = os.path.dirname(__file__)
    current_work_dir = os.path.join(root_dir, "process")
    _docs_path = os.path.join(current_work_dir, "final.txt")
    _param_pkl = os.path.join(current_work_dir, "final_param.plk")
    _stop_words_path = os.path.join(current_work_dir, "cn_stopwords.txt")

    # _docs_path = "./final.txt"
    # # path = ""
    # _param_pkl = "./final_param.plk"
    # _stop_words_path = "./cn_stopwords.txt"

    #     _stop_words_path = "data/stop_words.txt"
    _stop_words = []

    def __init__(self, docs=""):
        self.docs = docs
        self.param: BM25Param = self._load_param()

    def _load_stop_words(self):
        if not os.path.exists(self._stop_words_path):
            raise Exception(f"system stop words: {self._stop_words_path} not found")
        stop_words = []
        with open(self._stop_words_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _build_param(self):

        def _cal_param(reader_obj):
            f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
            df = {}  # 存储每个词及出现了该词的文档数量
            idf = {}  # 存储每个词的idf值
            lines = reader_obj.readlines()
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                words = [word for word in jieba.lcut(line) if word and word not in self._stop_words]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                idf[word] = math.log(length - num + 0.5) - math.log(num + 0.5)
            param = BM25Param(f, df, idf, length, words_count / length, docs_list, line_length_list)
            return param

        # cal
        if self.docs:
            if not os.path.exists(self.docs):
                raise Exception(f"input docs {self.docs} not found")
            with open(self.docs, 'r', encoding='utf8') as reader:
                param = _cal_param(reader)

        else:
            if not os.path.exists(self._docs_path):
                raise Exception(f"system docs {self._docs_path} not found")
            with open(self._docs_path, 'r', encoding='utf8') as reader:
                param = _cal_param(reader)

        with open(self._param_pkl, 'wb') as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self):
        self._stop_words = self._load_stop_words()
        if self.docs:
            param = self._build_param()
        else:
            if not os.path.exists(self._param_pkl):
                param = self._build_param()
            else:
                with open(self._param_pkl, 'rb') as reader:
                    param = pickle.load(reader)
        return param

    def _cal_similarity(self, words, index):
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            molecular = self.param.idf[word] * self.param.f[index][word] * (self.param.k1 + 1)
            denominator = self.param.f[index][word] + self.param.k1 * (1 - self.param.b +
                                                                       self.param.b * self.param.line_length_list[
                                                                           index] /
                                                                       self.param.avg_length)
            score += molecular / denominator
        return score

    def cal_similarity(self, query: str):
        """
        相似度计算，无排序结果
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        words = [word for word in jieba.lcut(query) if word and word not in self._stop_words]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))
        return score_list

    def cal_similarity_rank(self, query: str):
        """
        相似度计算，排序
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        result = self.cal_similarity(query)
        result.sort(key=lambda x: -x[1])
        return result


def process_input(test_json_path, test_csv_output_path):
    print("\n输入路径：{0}， \n输出路径：{1}".format(test_json_path, test_csv_output_path))
    stage1 = False
    if stage1:
        # 一阶段格式
        examples = pd.read_json(test_json_path, dtype=False)
        idx = []
        statement = []
        answer = []
        option_A = []
        option_B = []
        option_C = []
        option_D = []

        for id_ in examples['id']:
            idx.append(id_)

        for statement_ in examples['statement']:
            statement.append(statement_)

        for example in examples['option_list']:
            option_A.append(example['A'])
            option_B.append(example['B'])
            option_C.append(example['C'])
            option_D.append(example['D'])

        data_dict = {'id': idx, 'statement': statement, 'option_A': option_A, 'option_B': option_B, 'option_C': option_C,
                     'option_D': option_D,
                     }
        data_concat_answer = pd.DataFrame(data_dict)
    else:
        # 二阶段格式
        data_all = []
        idx = []
        statement = []
        option_A = []
        option_B = []
        option_C = []
        option_D = []

        with open(test_json_path, 'r', encoding='utf-8') as fpr:
            for f in fpr.readlines():
                data_raw = json.loads(f)
        #         print(data_raw)
                idx.append(data_raw['id'])
                statement.append(data_raw['statement'])
                option_A.append(data_raw['option_list']['A'])
                option_B.append(data_raw['option_list']['B'])
                option_C.append(data_raw['option_list']['C'])
                option_D.append(data_raw['option_list']['D'])
        data_dict = {'id': idx, 'statement': statement, 'option_A': option_A, 'option_B': option_B, 'option_C': option_C,
                     'option_D': option_D,
                     }
        data_concat_answer = pd.DataFrame(data_dict)


    print("\n开始处理文件：", test_json_path,)

    # 开始bm25查找
    bm_data = data_concat_answer

    bm_data_paire_a = bm_data['statement'] + " " + bm_data['option_A']
    bm_data_paire_b = bm_data['statement'] + " " + bm_data['option_B']
    bm_data_paire_c = bm_data['statement'] + " " + bm_data['option_C']
    bm_data_paire_d = bm_data['statement'] + " " + bm_data['option_D']

    bm25 = BM25()
    bm_text_a = []
    bm_text_b = []
    bm_text_c = []
    bm_text_d = []

    for q1, q2, q3, q4 in zip(tqdm.tqdm(bm_data_paire_a, leave=False), bm_data_paire_b, bm_data_paire_c, bm_data_paire_d):
        result1 = bm25.cal_similarity_rank(q1)
        result2 = bm25.cal_similarity_rank(q2)
        result3 = bm25.cal_similarity_rank(q3)
        result4 = bm25.cal_similarity_rank(q4)
        # 保存top1
        for (t1, score1), (t2, score2), (t3, score3), (t4, score4) in zip(result1, result2, result3, result4):
            bm_text_a.append(t1)
            bm_text_b.append(t2)
            bm_text_c.append(t3)
            bm_text_d.append(t4)
            break
    bm_dict = {'id': bm_data['id'], 'statement': bm_data['statement'],
               'option_A': bm_data['option_A'], 'option_B': bm_data['option_B'],
               'option_C': bm_data['option_C'], 'option_D': bm_data['option_D'],
               'context_a': bm_text_a, 'context_b': bm_text_b, 'context_c': bm_text_c, 'context_d': bm_text_d}

    bm_frame = pd.DataFrame(bm_dict)

    data_df = bm_frame
    data_df['ida'] = data_df['id'].astype('str') + 'a'
    data_df['idb'] = data_df['id'].astype('str') + 'b'
    data_df['idc'] = data_df['id'].astype('str') + 'c'
    data_df['idd'] = data_df['id'].astype('str') + 'd'
    #     print(data_train)

    data_test_a = pd.concat([data_df['ida'], data_df['statement'], data_df['option_A'], data_df['context_a']],
                            axis=1).rename(
        columns={'ida': 'id', 'option_A': 'option', 'context_a': 'context'})
    data_test_b = pd.concat([data_df['idb'], data_df['statement'], data_df['option_B'], data_df['context_b']],
                            axis=1).rename(
        columns={'idb': 'id', 'option_B': 'option', 'context_b': 'context'})
    data_test_c = pd.concat([data_df['idc'], data_df['statement'], data_df['option_C'], data_df['context_c']],
                            axis=1).rename(
        columns={'idc': 'id', 'option_C': 'option', 'context_c': 'context'})
    data_test_d = pd.concat([data_df['idd'], data_df['statement'], data_df['option_D'], data_df['context_d']],
                            axis=1).rename(
        columns={'idd': 'id', 'option_D': 'option', 'context_d': 'context'})

    test_data = pd.concat([data_test_a, data_test_b, data_test_c, data_test_d], axis=0)
    test_data.sort_index(axis=0,ascending=True, inplace=True)
    test_data.to_csv(test_csv_output_path, sep=',', index=False, header=True, encoding='utf_8_sig')
    print("原始数据数据处理完毕.......文件保存在", test_csv_output_path)


class Tasker():
    def __init__(self, model='roberta_wwm_ext_large', save_param_name='base', use_cnn=False, use_plus=False,
                 best_model=False):
        if use_cnn and use_plus:
            print(" use_cnn和use_plus只能使用一个，他们是不同的网络结构")
            return
        # 拼接路径所用的模型名称
        self.model_name = model
        self.base_model_name = model
        self.save_param_name = save_param_name
        # 自己搭建的网络结构
        self.net = Net
        if use_cnn:
            self.net = NetCNN
            self.model_name = model + '-cnn'
        if use_plus:
            self.net = NetPlus
            self.model_name = model + '-plus'
        self.best_model = best_model
        self.use_cnn = use_cnn
        self.trainer = None
        # 是否训练,用来区分compute_metrics里面的计算
        self.is_train = True
        # 初始化模型所有路径
        self.init_path()
        # 初始化数据和有关模型
        self.init_data_and_model()

    # 初始化数据和模型
    def init_data_and_model(self):
        if self.best_model:
            self.Datasets = CAILDataset(tokenizer_path=self.best_tokenizer_path, train_data_path=self.train_data_path,
                                        dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        else:
            self.Datasets = CAILDataset(tokenizer_path=self.tokenizer_path, train_data_path=self.train_data_path,
                                        dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        self.tokenizer = self.Datasets.get_tokenizer()
        # 动态填充，即将每个批次的输入序列填充到一样的长度
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='max_length')
        self.raw_datasets = self.Datasets.get_raw_datasets()
        # 加载模型
        if self.best_model:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            self.config = AutoConfig.from_pretrained(self.best_config_path, output_hidden_states=True)
            # 从保存的模型里面加载，包括了分类层的权重
            self.model = self.net.from_pretrained(pretrained_model_name_or_path=self.best_model_path,
                                                  config=self.config)
        else:
            # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
            self.config = AutoConfig.from_pretrained(self.config_path, output_hidden_states=True)
            # 从基模型里面加载，不包括分类层的权重
            self.model = self.net(config=self.config)
            # 初始化基模型的权重
            self.model.init_base_model(model_path=self.model_path)


    # 初始化路径
    def init_path(self):
        self.output_dir_path = "output/" + self.save_param_name + "/" + self.model_name
        self.logging_dir_path = "log/" + self.save_param_name + "/" + self.model_name
        self.save_model_path = "best_models/" + self.save_param_name + "/" + self.model_name

        self.model_path = "models/" + self.base_model_name
        self.tokenizer_path = "models/" + self.base_model_name + "/"
        self.config_path = "models/" + self.base_model_name + "/config.json"

        # best_path = "best_models/" + self.save_param_name + "/" + self.model_name
        output_model_checkpoint = "DeBERTa-v2-97M-Chinese"#/checkpoint-27000
        best_path = 'output/' + output_model_checkpoint
        self.best_model_path = best_path
        self.best_config_path = best_path + "/config.json"
        self.best_tokenizer_path = best_path + "/"

        self.train_data_path = "corpus/binary/train/train_top1.csv"
        self.dev_data_path = "corpus/binary/dev/dev_top1.csv"
        self.test_data_path = "corpus/binary/test/test_top1.csv"

    def get_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir_path,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=500,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=500,
            save_total_limit=25,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=128,
            num_train_epochs=20,
            weight_decay=1e-4,
            logging_dir=self.logging_dir_path,
            load_best_model_at_end=True,
            metric_for_best_model='absolute_accuracy',
            # no_cuda=True,
        )
        return training_args

    def train(self, use_fgm=False, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        self.is_train = True
        # 加载训练数据
        self.train_datasets = self.Datasets.LoadTrainDataset()
        # 加载验证数据
        self.dev_datasets = self.Datasets.LoadDevDataset()
        # 得到训练器的超参数
        training_args = self.get_training_args()

        early_stop = EarlyStoppingCallback(early_stopping_patience=10,
                                           early_stopping_threshold=0.00005
                                           )
        # 构造训练器
        self.trainer = MyTrainer(
            self.model,
            training_args,
            train_dataset=self.train_datasets,
            eval_dataset=self.dev_datasets,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            use_fgm=use_fgm,
            # callbacks=[early_stop],
            # optimizers=(optimizer, None)
        )
        # 训练模型
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        # 加载最优模型评估
        self.trainer.evaluate(self.dev_datasets)
        # 保存模型
        self.trainer.save_model(self.save_model_path)

    def test(self):
        """
        使用保存下来的最好的模型进行测试
        :return:
        """
        # 加载测试数据
        self.test_datasets = self.Datasets.LoadTestDataset()
        predict_args = TrainingArguments("temp",
                                         per_device_eval_batch_size=8,
                                         # no_cuda=True
                                         )
        # 构造训练器
        self.trainer = MyTrainer(
            self.model,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            # compute_metrics=self.compute_metrics,
            args=predict_args

        )

        path = os.path.dirname(__file__)
        submit_path = os.path.join(path, "temp")
        if not os.path.exists(submit_path):
            os.mkdir(submit_path)

        output_submit_path = os.path.join(submit_path, self.save_param_name)
        if not os.path.exists(output_submit_path):
            os.mkdir(output_submit_path)

        # output_submit_path_and_filename = path + '/result.txt'
        output_submit_path_and_filename = os.path.join(path, "result.txt")
        # row_output_submit_path_and_filename = output_submit_path + '/row_submit.txt'
        row_output_submit_path_and_filename = os.path.join(output_submit_path, 'row_submit.txt')
        # save_csv_path_and_filename = output_submit_path + '/output_labels.csv'
        save_csv_path_and_filename = os.path.join(output_submit_path, 'output_labels.csv')
        self.is_train = False
        idx = self.test_datasets["id"]
        logits, label_ids, metrics = self.trainer.predict(self.test_datasets, ignore_keys=['labels'])
        save_submition(logits=logits, index=idx, csv_path=save_csv_path_and_filename,
                       row_output_path=row_output_submit_path_and_filename,
                       output_path=output_submit_path_and_filename)

    def model_init(self):
        # 加载模型配置 output_hidden_states是否获取所有隐藏层的输出
        config = AutoConfig.from_pretrained(self.config_path, output_hidden_states=True)
        # 加载模型
        model = self.net(config=config)
        model.init_base_model(model_path=self.model_path)
        return model

    # 搜索超参数
    def search_hyperparameter_train(self):
        self.is_train = True
        Datasets = CAILDataset(tokenizer_path=self.tokenizer_path, train_data_path=self.train_data_path,
                               dev_data_path=self.dev_data_path, test_data_path=self.test_data_path)
        # 得到训练器的超参数
        training_args = self.get_training_args()
        self.tokenizer = Datasets.get_tokenizer()
        # 加载训练数据
        self.train_datasets = Datasets.LoadTrainDataset()
        # 加载验证数据
        self.dev_datasets = Datasets.LoadDevDataset()
        # 动态填充，即将每个批次的输入序列填充到一样的长度
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='longest')
        # padding = max_length
        # 构造训练器
        self.trainer = MyTrainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=self.train_datasets,
            eval_dataset=self.dev_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        best_run = self.trainer.hyperparameter_search(n_trials=10, compute_objective=accuracy_compute_objective,
                                                      direction="maximize")
        print(best_run)
        f = open(self.logging_dir_path + '_best_run.txt', 'w')
        print(best_run, file=f)
        print("*" * 200)
        # 设置超参数并训练
        for n, v in best_run.hyperparameters.items():
            setattr(self.trainer.args, n, v)
        self.trainer.train()
        # 保存模型
        self.trainer.save_model(self.save_model_path)
        return best_run

    # 打印模型参数
    def print_model(self):
        f = open(self.logging_dir_path + '_model_parameters.txt', 'w')
        f1 = open(self.logging_dir_path + '_struc.txt', 'w')
        # print(model, file=f)
        params = list(self.model.named_parameters())
        par, num = zip(*params)
        for name, parameters in zip(self.model.named_parameters(), self.model.parameters()):  # 打印出参数矩阵及值
            print(name, parameters, file=f)
        for name in par:  # 打印出参数矩阵及值
            print(name, file=f1)

    # 指标评估
    def compute_metrics(self, eval_predictions):

        predictions, label_ids = eval_predictions  # 概率-标签
        preds = np.argmax(predictions, axis=1)
        dev_idx = self.dev_datasets["id"]

        save_dev_file = False
        if save_dev_file:  # 全量数据训练

            path_row = os.getcwd()
            submit_path = path_row + '/temp'
            if not os.path.exists(submit_path):
                os.mkdir(submit_path)

            path = submit_path + '/' + self.save_param_name
            if not os.path.exists(path):
                os.mkdir(path)

            csv_path = path + '/out_dev.csv'
            row_path = path + '/row_dev.txt'
            add_path = path + '/out_dev.txt'
            save_submition(logits=predictions, index=dev_idx, csv_path=csv_path, row_output_path=row_path,
                           output_path=add_path)

        logits = torch.from_numpy(predictions)
        prob = F.softmax(logits, dim=1)
        pre_label = torch.argmax(prob, dim=1).numpy()
        true_label = label_ids
        probably = torch.max(prob, dim=1).values.numpy()

        # 将预测结果写入文件
        data = pd.DataFrame(data=[dev_idx, true_label, pre_label, probably],
                            index=['id', 'true_label', 'pre_label', 'probably']).T

        data_a = data[data['id'].str.contains('a')]
        data_b = data[data['id'].str.contains('b')]
        data_c = data[data['id'].str.contains('c')]
        data_d = data[data['id'].str.contains('d')]

        # 计算预测的正确率
        absolute_correct = 0
        add_correct = 0
        total = 0
        for a_true, b_true, c_true, d_true, a_pre, b_pre, c_pre, d_pre, a_pro, b_pro, c_pro, d_pro in zip(
                data_a['true_label'], data_b['true_label'], data_c['true_label'], data_d['true_label'],
                data_a['pre_label'], data_b['pre_label'], data_c['pre_label'], data_d['pre_label'],
                data_a['probably'], data_b['probably'], data_d['probably'], data_d['probably']):
            total = total + 1
            if a_true == a_pre and b_true == b_pre and c_true == c_pre and d_true == d_pre:
                absolute_correct = absolute_correct + 1

            if a_pre + b_pre + c_pre + d_pre == 0:
                if min(a_pro, b_pro, c_pro, d_pro) == a_pro:
                    a_pre = 1
                if min(a_pro, b_pro, c_pro, d_pro) == b_pro:
                    b_pre = 1
                if min(a_pro, b_pro, c_pro, d_pro) == c_pro:
                    c_pre = 1
                if min(a_pro, b_pro, c_pro, d_pro) == d_pro:
                    d_pre = 1
            if a_true == a_pre and b_true == b_pre and c_true == c_pre and d_true == d_pre:
                add_correct = add_correct + 1
        absolute_accuracy = float(absolute_correct / total)
        add_accuracy = float(add_correct / total)
        accuracy_metric = load_metric('./utils/accuracy.py')
        accuracy_result = accuracy_metric.compute(predictions=preds, references=label_ids)
        return {"absolute_accuracy": absolute_accuracy, "add_accuracy": add_accuracy,
                "accuracy": accuracy_result['accuracy']}


if __name__ == '__main__':

    '''
    
    # ===========数据处理=========
    test_input_name = "test_input2.json"
    test_output_name = 'test2.csv'

    root_path = os.path.dirname(__file__)
    corpus_path = os.path.join(root_path, "corpus")
    row_data_path = os.path.join(corpus_path, "row_data")
    test_input_path = os.path.join(row_data_path, test_input_name)
    test_output_path = os.path.join(corpus_path, test_output_name)

    if not os.path.exists(test_output_path):  # 处理好的文件不存在
        process_input(test_json_path=test_input_path, test_csv_output_path=test_output_path)
    '''
    # ===========运行模型===========
    os.environ["WANDB_DISABLED"] = "true"
    save_param_name = 'main_all_sort_train'
    model = 'DeBERTa-v2-97M-Chinese'
    # tasker = Tasker(model=model, save_param_name=save_param_name, use_cnn=False, use_plus=False, best_model=False)
    # tasker.train(use_fgm=False, resume_from_checkpoint=False)
    # tasker.print_model()

    # tasker.search_hyperparameter_train()

    # 预测
    tasker = Tasker(model=model, save_param_name=save_param_name, use_cnn=False, use_plus=False, best_model=True)
    tasker.test()
