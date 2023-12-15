import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
import os
from func_module import freqoracle
from func_module import errormetric
from func_module import realfreq
import copy


# 用于计算参数theta
def theta_calculation(ahead_tree_height, epsilon, user_scale, branch):
    # ahead_tree_height：树的高度，整数型
    # epsilon：隐私预算，浮点型
    # user_scale：用户规模，整数型
    # branch：分支数量，整数型，默认为2
    user_scale_in_each_layer = user_scale / ahead_tree_height  # 利用用户的总数量除以ahead树的高度来计算每一层的用户数量
    varience_of_OUE = 4 * math.exp(epsilon) / (
                user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)  # 每一层利用OUE所产生的误差
    return math.sqrt((branch + 1) * varience_of_OUE)  # 根据理论得到是否需要划分的参数theta


def theta1_calculation(ahead_tree_height, epsilon, user_scale, branch, d):
    # ahead_tree_height：树的高度，整数型
    # epsilon：隐私预算，浮点型
    # user_scale：用户规模，整数型
    # branch：分支数量，整数型，默认为2
    user_scale_in_each_layer = user_scale / ahead_tree_height  # 利用用户的总数量除以ahead树的高度来计算每一层的用户数量
    varience_of_GRR = (math.exp(epsilon) + d - 2) / (
                user_scale_in_each_layer * (math.exp(epsilon) - 1) ** 2)  # 每一层利用OUE所产生的误差
    return math.sqrt((branch + 1) * varience_of_GRR)  # 根据理论得到是否需要划分的参数theta


def construct_translation_vector(domain_size, branch):
    # domain_size：数据主域大小，整形
    # branch：分支数量，整数型，默认为2
    translation_vector = []
    for i in range(branch):
        translation_vector.append(np.array(
            [i * domain_size // branch, i * domain_size // branch]))
    return translation_vector



# 移除重复的子域划分向量
def duplicate_remove(list1):
    # list1：多维数组
    list2 = []
    for li1 in list1:
        Flag1 = True  # 用于判断是否将li1添加到list2

        for li2 in list2:
            if (li1 == li2).all():  # 如果两个数组里的元素全部相等，结果为真，若有一个不相等，结果为假
                Flag1 = False  # 如果两个数组元素完全相等，置flag为false，那么就break，之后的if判断就不会把li1添加进list2
                break

        if Flag1 == True:
            list2.append(li1)
    return list2


# Step1: User Partition (UP) in Section 4.2
def user_record_partition(data_path, ahead_tree_height, domain_size):
    # data_path：数据路径
    # ahead_tree_height：ahead树的高度
    # 主域的大小
    dataset = np.loadtxt(data_path, np.int64)  # 从文本加载数据
    user_sample_id = np.random.randint(0, ahead_tree_height, len(dataset)).reshape(len(dataset),
                                                                                   1)  # 先是生成从0到ahead树高度之间的随机整数，数量为数据集的大小的数组，然后把他们按照每列一个排列
    user_histogram = np.zeros((ahead_tree_height, domain_size),
                              dtype=np.int32)  # 返回一个类型为int型，行数为ahead_tree_height，列数为domain_size大小的二维数组
    for k, item in enumerate(dataset):
        user_histogram[user_sample_id[k], item] += 1  # 将用户数据随机分到每个组里
    return user_histogram  # 返回的是一个二维数组，行为数据主域的大小，列为树的高度，那么每行对应元素的含义是该组存在此元素的数量，比如user_histogram[0][0]就是第0组数据为0的用户的数量






def Optimized_ahead_tree_update(ahead_tree, tree_height, theta, branch, translation_vector, layer_index, data_size):
    num = len(ahead_tree.leaves())

    theta1 = theta1_calculation(optimized_ahead_tree_height, epsilon, data_size, branch, num)  # 计算阈值theta2

    for node in ahead_tree.leaves():
        if not node.data.divide_flag:
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (
                num < (3 * math.exp(epsilon) + 2) and node.data.frequency < theta1):
            node.data.divide_flag = False
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (
                num >= (3 * math.exp(epsilon) + 2) and node.data.frequency < theta):
            node.data.divide_flag = False
            continue
        else:

            TempItem0 = np.zeros(node.data.interval.shape)  # 生成值为0大小为2的一维数组，存放区间的开始与结束地址，未重定位，比如第一个区间是[0,512]
            for j in range(0, len(node.data.interval), 2):
                if node.data.interval[j + 1] - node.data.interval[j] > 1:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch + \
                                       node.data.interval[j]
                else:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = node.data.interval[j + 1]
            # translation_vector 存放区间的偏移地址，比如第一个未[(0,0),(512,512)],另它的每个元素加上第一个TempItem0也就是[0,512]就得到最终的第一次分解区间（0，512）与（512，1024）
            for item1 in translation_vector:
                node_name = str(tree_height) + str(layer_index)
                node_frequency = 0
                node_divide_flag = True
                node_count = 0

                node_interval = TempItem0 + item1

                ahead_tree.create_node(node_name, node_name, parent=node.identifier,
                                       data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
                layer_index += 1





def Optimized_ahead_tree_construction(optimized_ahead_tree, ahead_tree_height, theta, branch, translation_vector,
                                    user_dataset_partition, data_size):
    tree_height = 0
    while tree_height < ahead_tree_height:
        layer_index = 0
        # update ahead_tree structrue

        Optimized_ahead_tree_update(optimized_ahead_tree, tree_height, theta, branch, translation_vector, layer_index, data_size)
        # update sub-domain partition vectors
        translation_vector[:] = translation_vector[:] // np.array([branch, branch])

        translation_vector = duplicate_remove(translation_vector)
        # update ahead_tree sub-domain frequency
        Optimized_node_frequency_aggregation(optimized_ahead_tree, user_dataset_partition[tree_height], epsilon)

        tree_height += 1







def Optimized_HIO_tree_construction(HIO_tree, HIO_tree_height, HIO_branch, translation_vector, user_dataset_partition):
    tree_height = 0
    while tree_height < HIO_tree_height:
        layer_index = 0
        # update ahead_tree structrue
        HIO_tree_update(HIO_tree, tree_height, HIO_branch, translation_vector, layer_index)
        # update sub-domain partition vectors
        translation_vector[:] = translation_vector[:] // np.array([HIO_branch, HIO_branch])

        translation_vector = duplicate_remove(translation_vector)
        # update ahead_tree sub-domain frequency
        Optimal_HIO_node_frequency_aggregation(HIO_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1


# Step4: Post-processing (PP) in Section 4.2
def ahead_tree_postprocessing(optimized_ahead_tree):
    lowest_nodes_number = 0

    for _, node in reversed(list(enumerate(optimized_ahead_tree.all_nodes()))):

        if lowest_nodes_number < optimized_ahead_tree.size(optimized_ahead_tree.depth()):  # .size()返回一层的所有节点  .depth()返回树的深度
            lowest_nodes_number += 1
            continue

        if optimized_ahead_tree.depth(node) != optimized_ahead_tree.depth() and optimized_ahead_tree.children(
                node.identifier) != []:  # 判断时候是否为最后一层且是否存在孩子节点

            numerator = 1 / node.data.count
            children_frequency_sum = 0
            for j, child_node in enumerate(optimized_ahead_tree.children(node.identifier)):
                numerator += 1 / child_node.data.count
                children_frequency_sum += child_node.data.frequency

            denominator = numerator + 1
            coeff0 = numerator / denominator
            coeff1 = 1 - coeff0

            node.data.frequency = coeff0 * node.data.frequency + coeff1 * children_frequency_sum
            node.data.count = 1 / coeff0





# answer range queries
def optimized_ahead_tree_answer_query(ahead_tree, query_interval, domain_size):
    # Ahead_tree:后处理过后的ahead树
    # query_interval：大小为2的数组，代表某个查询区间，如[a,b]
    # domain_size:数据主域大小
    estimated_frequency_value = 0
    # set 1-dim range query
    query_interval_temp = np.zeros(domain_size)
    d1_left = int(query_interval[0])
    d1_right = int(query_interval[1])
    query_interval_temp[d1_left:d1_right] = 1
    for i, node in enumerate(ahead_tree.all_nodes()):
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])

        if query_interval_temp.sum() and ahead_tree.children(node.identifier) != [] and query_interval_temp[
                                                                                        d1_left:d1_right].sum() == (
                d1_right - d1_left):
            estimated_frequency_value = estimated_frequency_value + node.data.frequency
            query_interval_temp[d1_left:d1_right] = 0
            continue

        if query_interval_temp.sum() and ahead_tree.children(node.identifier) == []:
            coeff = query_interval_temp[d1_left:d1_right].sum() / (d1_right - d1_left)
            estimated_frequency_value = estimated_frequency_value + coeff * node.data.frequency
            query_interval_temp[d1_left:d1_right] = 0

    return estimated_frequency_value





# record query errors
def optimized_ahead_tree_query_error_recorder(optimized_ahead_tree, real_frequency, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    for i, query_interval in enumerate(query_interval_table):
        d1_left = int(query_interval[0])
        d1_right = int(query_interval[1])
        real_frequency_value = real_frequency[i]
        estimated_frequency_value = optimized_ahead_tree_answer_query(optimized_ahead_tree, query_interval, domain_size)
        errList[i] = real_frequency_value - estimated_frequency_value
        print('answer index {}-th query'.format(i))
        print("real_frequency_value: ", real_frequency_value)
        print("estimated_frequency_value: ", estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))





def node_frequency_aggregation(ahead_tree, user_dataset, epsilon):
    # ahead_tree:ahead树
    # user_dataset:高度为hight层的所有用户数据
    # epsilon：隐私预算
    # estimate the frequency values, and update the frequency values of the nodes
    p = 0.5  # OUE的概率参数选择
    q = 1.0 / (1 + math.exp(epsilon))  # OUE的概率参数选择

    user_record_list = []
    for node in ahead_tree.leaves():
        d1_left = int(node.data.interval[0])

        d1_right = int(node.data.interval[1])

        user_record_list.append(user_dataset[d1_left:d1_right].sum())  # 统计这个分组区间内的用户数量
    print("user_record_list", len(user_record_list))
    print(len(ahead_tree.leaves()))
    noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32),
                                        sum(user_record_list))  # 调用OUE算法计算噪音矩阵,也就是加了噪音后的数据量(还没纠正噪音)
    noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p,
                                          q)  # 返回经过Norm_Sub后处理过的频率

    for i, node in enumerate(ahead_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (
                        node.data.count + 1)  # 这一步它对于没有被划分的区间用多次频率取平均(原文好像没有)
            node.data.count += 1


def Optimized_node_frequency_aggregation(optimized_ahead_tree, user_dataset, epsilon):
    # ahead_tree:ahead树
    # user_dataset:高度为hight层的所有用户数据
    # epsilon：隐私预算
    # estimate the frequency values, and update the frequency values of the nodes
    p_OUE = 0.5  # OUE的概率参数选择
    q_OUE = 1.0 / (1 + math.exp(epsilon))  # OUE的概率参数选择

    user_record_list = []
    print(len(optimized_ahead_tree.leaves()))
    for node in optimized_ahead_tree.leaves():
        d1_left = int(node.data.interval[0])

        d1_right = int(node.data.interval[1])

        user_record_list.append(user_dataset[d1_left:d1_right].sum())  # 统计这个分组区间内的用户数量
    print("user_record_list", len(user_record_list))
    if (len(user_record_list) < 3 * math.exp(epsilon) + 2):
        p_GRR = math.exp(epsilon) / (math.exp(epsilon) + len(user_record_list) - 1)
        q_GRR = 1 / (math.exp(epsilon) + len(user_record_list) - 1)
        noise_vector = freqoracle.GRR_Noise(epsilon, np.array(user_record_list, np.int32), sum(user_record_list))
        noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p_GRR,
                                              q_GRR)  # 返回经过Norm_Sub后处理过的频率
    else:
        noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32),
                                            sum(user_record_list))  # 调用OUE算法计算噪音矩阵,也就是加了噪音后的数据量(还没纠正噪音)
        noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p_OUE,
                                              q_OUE)  # 返回经过Norm_Sub后处理过的频率

    for i, node in enumerate(optimized_ahead_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (
                        node.data.count + 1)  # 这一步它对于没有被划分的区间用多次频率取平均(原文好像没有)
            node.data.count += 1





# 新版的treelib带的功能，定义一个类存放数据
class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval):
        self.frequency = frequency  # 频率
        self.divide_flag = divide_flag  # 是否需要划分的标签
        self.count = count
        self.interval = interval  # 所代表的主域区间






def Optimized_AHEAD_main_func(repeat_time, domain_size, branch, ahead_tree_height, theta, real_frequency,
                            query_interval_table, epsilon, data_path, data_name, data_size_name, domain_name,
                            data_size):
    MSEDict = {'rand': []}
    print("MSEDict", MSEDict)
    repeat = 0
    while repeat < repeat_time:
        # user partition
        user_dataset_partition = user_record_partition(data_path, ahead_tree_height,
                                                       domain_size)  # 返回的是一个二维数组，每行包括用户的数据，列为树的高度

        # initialize the tree structure, set the root node
        optimized_ahead_tree = Tree()  # 初始化ahead树
        optimized_ahead_tree.create_node('Root', 'root',
                               data=Nodex(1, True, 1, np.array([0, domain_size])))  # 创建一个根节点为Root，存储的数据为Nodex的树

        # construct sub-domain partition vectors
        translation_vector = construct_translation_vector(domain_size, branch)

        # build a tree structure
        Optimized_ahead_tree_construction(optimized_ahead_tree, ahead_tree_height, theta, branch, translation_vector,
                                        user_dataset_partition, data_size)

        # ahead_tree post-processing
        ahead_tree_postprocessing(optimized_ahead_tree)

        # ahead_tree answer query
        optimized_ahead_tree_query_error_recorder(optimized_ahead_tree, real_frequency, query_interval_table, domain_size,
                                        MSEDict)  # 将查询结果的均方误差返回字典里

        # record errors

        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')  # pandas中的对数据处理工具

        MSEDict_temp.to_csv('rand_result/MSE_lle_Optimal_ahead_branch{}-{}-{}-{}-{}-{}.csv'.format(branch,
                                                                                                   data_name,
                                                                                                   data_size_name,
                                                                                                   domain_name,
                                                                                                   epsilon,
                                                                                                   repeat_time))
        repeat += 1
        print("repeat time: ", repeat)








if __name__ == "__main__":

    # 重复的实验次数
    repeat_time = 1

    # 设置隐私预算
    epsilon = 1
    # 设置数据维度，树的分支和数据主域大小
    data_dimension = 1
    branch = 2

    domain_size = 2 ** 10
    optimized_ahead_tree_height = int(math.log(domain_size, branch))  # 计算ahead树的高度

    # load query table
    query_path = './query_table/rand_query_domain10_attribute{}.txt'.format(data_dimension)
    query_interval_table = np.loadtxt(query_path, int)  # 存储要查询的所有区间，每行都是两个元素，第一个元素代表区间的开始，最后一个元素代表区间的结束
    print("the top 5 range queries in query_interval_table: \n", query_interval_table[:5])

    # select dataset
    data_name = '1dim_normal'
    data_size_name = 'set_10_5'
    domain_name = 'domain10_attribute{}'.format(data_dimension)

    # load dataset
    data_path = './dataset/{}-{}-{}-data.txt'.format(data_name, data_size_name, domain_name)
    dataset = np.loadtxt(data_path, np.int32)
    print("the shape of dataset: ", dataset.shape)
    data_size = dataset.shape[0]  # 数据的行数

    # calculate/load true frequency
    real_frequency_path = './query_table/real_frequency-{}-{}-{}.npy'.format(data_name, data_size_name, domain_name)
    if os.path.exists(real_frequency_path):
        real_frequency = np.load(real_frequency_path)
    else:
        real_frequency = realfreq.real_frequency_generation(dataset, data_size, domain_size, data_dimension,
                                                            query_interval_table)
        np.save(real_frequency_path, real_frequency)



# Optimal_AHEAD-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    theta = theta_calculation(optimized_ahead_tree_height, epsilon, data_size, branch) #计算阈值theta
    Optimized_AHEAD_main_func(repeat_time, domain_size, branch, optimized_ahead_tree_height, theta, real_frequency, query_interval_table, epsilon,data_path, data_name, data_size_name, domain_name, data_size)
# Optimal_AHEAD-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

