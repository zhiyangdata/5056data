import networkx as nx
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import tensorflow as tf
import math
import time
from alias import alias_sample, create_alias_table
class DBLPDataLoader:
    def __init__(self, graph_file,g=None):
        #创建图
        if(g ==None):

            self.g = nx.read_edgelist(graph_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
        else:
            self.g=g
        #self.g = nx.read_edgelist(graph_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

        #边和点的数量
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()

        #边和点的集合
        self.edges_raw = self.g.edges()
        self.nodes_raw = self.g.nodes()
        self.embedding = []#嵌入向量
        self.neg_nodeset={}
        #为每一个node维护一个负采样群
        self.neg_proportion={}
        #self.neg_nodeset =[]#负采样点集
        self.node_index = {}#节点索引
        self.node_index_reversed = {}
        for index, node in enumerate(self.nodes_raw):
            self.node_index[node] = index#{node：index}
            self.node_index_reversed[index] = node#{index:node}
        #将边转化为index形式
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges_raw]
        #按照节点原本的度
        node_degrees=[val for (node,val) in self.g.degree()]
        self.node_degree = node_degrees
        node_degrees /= np.sum(node_degrees)
        self.node_degree_distribution11=node_degrees
        #按照3/4度进行采样
        node34_degrees=np.power(self.node_degree, 3 / 4)
        node34_degrees /= np.sum(node34_degrees)
        self.node_degree_distribution34 = node34_degrees
        print(self.g.number_of_edges())
        self.preprocess_transition_probs()
        for index in self.node_index.values():
            #print("yunx")
            self.neg_proportion[index]=0
        #按照每个节点度分布概率采样

    def deepwalk_walk(self, walk_length):
        start_node = np.random.choice(self.nodes_raw)#从node中随机选择节点
        walk = [start_node]
        walk_index = [self.node_index[start_node]]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.g.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk_node = np.random.choice(cur_nbrs)
                walk.append(walk_node)
                walk_index.append(self.node_index[walk_node])
            else:
                break
        return walk_index#只能返回一条游走序列[index1,index2,...,index_walk_length]
#设置两个K，一个为种群数量，一个为neg_node数量

    #node2vec采样
    def sigmoid1(x):
        return 1. / (1 + np.exp(-x))



    def node2vec_walk(self, walk_length):


        G = self.g
        start_node = np.random.choice(G.nodes())
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges


        walk = [start_node]
        walk_index = [self.node_index[start_node]]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk_node=cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                    walk.append(walk_node)
                    walk_index.append(self.node_index[walk_node])
                else:


                    prev = walk[-2]
                    edge = (prev, cur)
                    if alias_edges.__contains__(edge):


                        next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                          alias_edges[edge][1])]
                        walk.append(next_node)
                        walk_index.append(self.node_index[next_node])
                    else:
                        edge=(cur,prev)
                        next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]
                        walk.append(next_node)
                        walk_index.append(self.node_index[next_node])



                    # next_node = cur_nbrs[alias_sample(prob = alias_edges[edge][0], alias=alias_edges[edge][1])]
                    # walk.append(next_node)
                    # walk_index.append(self.node_index[next_node])

            else:
                break

        return walk_index
    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.g
        p = 1
        q = 1

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.g

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            u=edge[0]
            v=edge[1]
            edge2=(v,u)
            alias_edges[edge2]=self.get_alias_edge(v,u)
        print("运行成功")

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return



    def fetch_batch(self, embedding, method_name=4,benchmark="DeepWalk", batch_size=8, K=100,k_2=10, window_size=2, walk_length=8,amax=0.225,a=2,t=0,T=2600):
        self.embedding = embedding

        u_i = []
        u_j = []
        label = []
        embedding_dim = embedding.shape[1]

        for i in range(batch_size):#对于每一个batch
            #每一个batch生成一个句子
            if benchmark=="DeepWalk":
                self.walk_index = self.deepwalk_walk(walk_length)
            else:
                self.walk_index=self.node2vec_walk(walk_length)
            #随机选择一个节点生成一个句子

            for index, node in enumerate(self.walk_index):
                #walk_index=[node0,node1,node2,node3,node4,node5,node6,node7]
                for n in range(max(index-window_size, 0), min(index+window_size+1, len(self.walk_index))):
                    #如果index=0,window_size=2,walk_length=8.
                    # 此时n=range(0,3)即n=0,1,2
                    #u_i里边有[node0,node0]
                    #u_j里边有[node1,node2]
                    #label里有[1,1]
                    if n != index:
                        u_j.append(self.walk_index[n])
                        u_i.append(node)

                        label.append(1.)


                #self.neg_nodeset = []
                u_one_hot = np.zeros(self.num_of_nodes)
                u_one_hot[node] = 1
                u_i_embedding = np.matmul(u_one_hot, self.embedding)

                # for node_neg in self.node_index.values():
                #     if node_neg not in self.walk_index:
                #         #有除了walk里所有的单词
                #         self.neg_nodeset.append(node_neg)

                t1=time.time()

                if int(method_name)==1:
                    #self.neg_nodeset={}
                    if node not in self.neg_nodeset.keys():
                        #tmp为整个图节点-句子中节点
                        tmp=[]
                        node_neg_init_list=[]

                        #tmp=list(set(self.node_index.values()).difference(set(self.walk_index)))
                        #node_neg_init_list=list(np.random.choice(tmp,size=K,replace=False))
                        node_neg_init_list=list(np.random.choice(list(self.node_index.values()),size=K,p=self.node_degree_distribution11,replace=False))
                        self.neg_nodeset[node]=node_neg_init_list
                        #print(str(node)+'has been init')
                    else:
                        #print(str(node)+'Negative sampling is iterating ')
                        tmp=[]
                        node_neg_init_list=[]
                        neg_one_hot=np.zeros((len(self.neg_nodeset[node]),self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][self.neg_nodeset[node][b]]=1
                        negnode_embedding=np.matmul(neg_one_hot,self.embedding)
                        #neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)/\
                                           (np.sqrt(np.sum(negnode_embedding*negnode_embedding,axis=1))*np.sqrt(np.sum(u_i_embedding*u_i_embedding)))
                        neg_node_distance = np.exp(-1*neg_node_distance) / np.sum(np.exp(-1*neg_node_distance))
                        #更新三者的比例
                        if self.neg_proportion[node]<amax:
                            self.neg_proportion[node]=amax*tf.sigmoid(a*t/T+2)
                        top_k=int(K*self.neg_proportion[node])
                        #优秀比例负样本
                        top_k_index=neg_node_distance.argsort()[::-1][0:top_k]
                        for i in top_k_index:
                            node_neg_init_list.append(self.neg_nodeset[node][i])
                        #将优秀样本的邻居也加入进来
                        #加入的邻居节点的数目
                        s_neig=2*top_k
                        tmp_s_node=[]
                        for s_node in node_neg_init_list:
                            tmp_ever_node=[self.node_index[u] for u in self.g.neighbors(self.node_indexs_node)]
                            tmp_s_node.extend(tmp_ever_node)
                        s_node=list((set(s_node).difference(set(self.walk_index))))
                        s_node_index=np.random.choice(s_node,size=s_neig)

                        #根据节点度的3/4来采样剩余的节点
                        #最后剩余的节点集合
                        last_node_list=list(set(self.node_index)-set(self.walk_index))
                        last_node_degree=[self.g.degree[i1] for i1 in last_node_list ]
                        last_node_degree=np.power(last_node_degree,3/4)
                        last_node_distribution=last_node_degree/np.sum(last_node_degree)
                        last_node_index=np.random.choice(last_node_list,size=K-top_k_index-s_neig,p=last_node_distribution)

                        #和并负采样集
                        node_neg_init_list.extend(s_node_index)
                        node_neg_init_list.extend(last_node_index)




                        #tmp里为整个图节点-walk里的节点-上一轮已经选择的neg_node
                        #tmp = list((set(self.node_index.values()).difference(set(self.walk_index))).difference(self.neg_nodeset[node]))
                        #剩余节点随机选择
                        #node_neg_init_list.extend(list(np.random.choice(tmp,size=K-top_k,replace=False)))
                        #node_neg_init_list.extend(list(np.random.choice(list(self.node_index.values()),size=K-top_k,p=self.node_degree,replace=False)))
                        self.neg_nodeset[node]=node_neg_init_list
                    sample_neg = list(np.random.choice(self.neg_nodeset[node], k_2, replace=False))
                elif int(method_name)==2:
                    #self.neg_nodeset = {}
                    if node not in self.neg_nodeset.keys():
                        # tmp为整个图节点-句子中节点
                        tmp = []
                        node_neg_init_list = []
                        self.neg_proportion[node] = 0.0
                        tmp = list(set(self.node_index.values()).difference(set(self.walk_index)))
                        node_neg_init_list=list(np.random.choice(tmp,size=K,replace=False))
                        # node_neg_init_list = list(
                        #     np.random.choice(list(self.node_index.values()), size=K, p=self.node_degree, replace=False))
                        self.neg_nodeset[node] = node_neg_init_list
                        # print(str(node)+'has been init')
                    else:
                        # print(str(node)+'Negative sampling is iterating ')
                        tmp = []
                        node_neg_init_list = []
                        neg_one_hot = np.zeros((len(self.neg_nodeset[node]), self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][self.neg_nodeset[node][b]] = 1
                        negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                        # neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance = np.sum(u_i_embedding * negnode_embedding, axis=1) / \
                                            (np.sqrt(np.sum(negnode_embedding * negnode_embedding, axis=1)) * np.sqrt(
                                                np.sum(u_i_embedding * u_i_embedding)))
                        neg_node_distance = np.exp(-1 * neg_node_distance) / np.sum(np.exp(-1 * neg_node_distance))

                        if self.neg_proportion[node] < amax:
                            self.neg_proportion = amax * tf.sigmoid(a * t / T + 2)
                        top_k = int(K * self.neg_proportion)  # 优秀比例负样本
                        top_k_index = neg_node_distance.argsort()[::-1][0:top_k]
                        for i in top_k_index:
                            node_neg_init_list.append(self.neg_nodeset[node][i])
                        # 剩余K-top_k个样本随机选择
                        # tmp里为整个图节点-walk里的节点-上一轮已经选择的neg_node
                        tmp = list((set(self.node_index.values()).difference(set(self.walk_index))).difference(
                            self.neg_nodeset[node]))
                        # 剩余节点随机选择
                        node_neg_init_list.extend(list(np.random.choice(tmp,size=K-top_k,replace=False)))
                        # node_neg_init_list.extend(list(
                        #     np.random.choice(list(self.node_index.values()), size=K - top_k, p=self.node_degree,
                        #                      replace=False)))
                        self.neg_nodeset[node] = node_neg_init_list

                        self.neg_nodeset[node] = node_neg_init_list
                        neg_one_hot = np.zeros((len(self.neg_nodeset[node]), self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][self.neg_nodeset[node][b]] = 1
                        negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                        # neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance = np.sum(u_i_embedding * negnode_embedding, axis=1) / \
                                            (np.sqrt(np.sum(negnode_embedding * negnode_embedding, axis=1)) * np.sqrt(
                                                np.sum(u_i_embedding * u_i_embedding)))
                        # neg_node_distance/=np.sum(neg_node_distance)
                        neg_node_distance = np.exp(-1 * neg_node_distance) / np.sum(np.exp(-1 * neg_node_distance))
                    sample_neg = list(np.random.choice(self.neg_nodeset[node], k_2,p=neg_node_distance, replace=False))
                elif int(method_name)==3:
                    #方法1指的是在采集负样本时从种群中随机采集，在更新种群是按照[优秀样本，优秀样本邻居，随机抽取]
                    #方法2指的是在种群内更新时，只保留优秀负样本，其他随机更新[优秀样本，随机抽取]。在采集负样本时随机采集
                    #方法3指的是以节点度的3/4采样
                    #方法4指的是更新种群是按照[优秀样本，优秀样本邻居，随机抽取]，并且在抽取负样本时按照其与正样本的距离进行抽取

                    sample_neg= list(
                        np.random.choice(list(self.node_index.values()), size=k_2, p=self.node_degree_distribution34))
                elif int(method_name)==4:
                    #self.neg_nodeset = {}
                    if node not in self.neg_nodeset.keys():
                        # tmp为整个图节点-句子中节点
                        tmp = []
                        node_neg_init_list = []
                        #print("test123")
                        #self.neg_proportion[node] = 0.0
                        # tmp=list(set(self.node_index.values()).difference(set(self.walk_index)))
                        # node_neg_init_list=list(np.random.choice(tmp,size=K,replace=False))
                        node_neg_init_list = list(
                            np.random.choice(list(self.node_index.values()), size=K, p=self.node_degree_distribution11,
                                             replace=False))
                        self.neg_nodeset[node] = node_neg_init_list
                        #self.neg_nodeset[node] = node_neg_init_list
                        neg_one_hot = np.zeros((len(self.neg_nodeset[node]), self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][self.neg_nodeset[node][b]] = 1
                        negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                        # neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance = np.sum(u_i_embedding * negnode_embedding, axis=1) / \
                                            (np.sqrt(np.sum(negnode_embedding * negnode_embedding, axis=1)) * np.sqrt(
                                                np.sum(u_i_embedding * u_i_embedding)))
                        # neg_node_distance/=np.sum(neg_node_distance)
                        neg_node_distance = np.exp( neg_node_distance) / np.sum(np.exp(neg_node_distance))


                        # print(str(node)+'has been init')
                    else:
                        print(str(node)+'Negative sampling is iterating ')
                        tmp = []
                        node_neg_init_list = []
                        neg_one_hot = np.zeros((len(self.neg_nodeset[node]), self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][int(self.neg_nodeset[node][b])] = 1
                        negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                        # neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance = np.sum(u_i_embedding * negnode_embedding, axis=1) / \
                                            (np.sqrt(np.sum(negnode_embedding * negnode_embedding, axis=1)) * np.sqrt(
                                                np.sum(u_i_embedding * u_i_embedding)))
                        neg_node_distance = np.exp(neg_node_distance) / np.sum(np.exp(neg_node_distance))
                        # 更新三者的比例
                        if self.neg_proportion[node] < amax:
                            self.neg_proportion[node] = amax * (1. / (1 + np.exp(-(a * t / (T + 2)))))
                        #print(self.neg_proportion[node])
                        top_k = int(K * self.neg_proportion[node])  # 优秀比例负样本
                        top_k_index = neg_node_distance.argsort()[::-1][0:top_k]
                        for i in top_k_index:
                            node_neg_init_list.append(self.neg_nodeset[node][i])
                        # 将优秀样本的邻居也加入进来
                        # 加入的邻居节点的数目
                        s_neig = 2 * top_k
                        # print(top_k)
                        # print(s_neig)
                        tmp_s_node = []

                        #tmp_ever_node = []
                        for s_node in node_neg_init_list:
                            tmp_ever_node = []
                            u=self.g.neighbors(self.node_index_reversed[int(s_node)])
                            if u is not None:
                                for i in u:
                                    tmp_ever_node.append(self.node_index[i])
                            tmp_s_node.extend(tmp_ever_node)
                        s_node = list((set(tmp_s_node).difference(set(self.walk_index))))
                        if len(s_node)>s_neig:
                            s_node_index = np.random.choice(s_node, size=s_neig)
                        else:
                            s_node_index=s_node
                            s_neig=len(s_node)

                        # 根据节点度的3/4来采样剩余的节点
                        # 最后剩余的节点集合
                        last_node_list = list(set(self.node_index) - set(self.walk_index))
                        print(len(last_node_list))
                        last_node_degree = [self.g.degree[self.node_index_reversed[int(i1)]] for i1 in last_node_list]
                        last_node_degree = np.power(last_node_degree, 3 / 4)
                        last_node_distribution = last_node_degree / np.sum(last_node_degree)
                        print(len(last_node_distribution))
                        last_node_index = np.random.choice(last_node_list, size=K - top_k - s_neig,
                                                           p=last_node_distribution)
                        # 和并负采样集

                        node_neg_init_list.extend(s_node_index)
                        node_neg_init_list.extend(last_node_index)

                        # tmp里为整个图节点-walk里的节点-上一轮已经选择的neg_node
                        # tmp = list((set(self.node_index.values()).difference(set(self.walk_index))).difference(self.neg_nodeset[node]))
                        # 剩余节点随机选择
                        # node_neg_init_list.extend(list(np.random.choice(tmp,size=K-top_k,replace=False)))
                        # node_neg_init_list.extend(list(np.random.choice(list(self.node_index.values()),size=K-top_k,p=self.node_degree,replace=False)))
                        #已更新的节点集

                        self.neg_nodeset[node] = node_neg_init_list
                        neg_one_hot = np.zeros((len(self.neg_nodeset[node]), self.num_of_nodes))
                        for b in range(len(self.neg_nodeset[node])):
                            neg_one_hot[b][int(self.neg_nodeset[node][b])] = 1
                        negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                         # neg_node_distance=np.sum(u_i_embedding*negnode_embedding,axis=1)
                        neg_node_distance = np.sum(u_i_embedding * negnode_embedding, axis=1) / \
                                            (np.sqrt(np.sum(negnode_embedding * negnode_embedding, axis=1)) * np.sqrt(
                                             np.sum(u_i_embedding * u_i_embedding)))
                        #neg_node_distance/=np.sum(neg_node_distance)
                        neg_node_distance = np.exp(neg_node_distance) / np.sum(np.exp(neg_node_distance))


                        #按照节点的相似度采样
                    #按照负采样种群中正节点与负节点的相似度采样
                    sample_neg = list(np.random.choice(self.neg_nodeset[node], k_2, p=neg_node_distance,replace=False))
                    #sample_neg = list(np.random.choice(self.neg_nodeset[node], k_2,  replace=False))
                else :
                    #按照节点度的分布采样
                    sample_neg = list(
                        np.random.choice(list(self.node_index.values()), size=k_2, p=self.node_degree_distribution11))

                # t2=time.time()
                # neg_time=t2-t1
                #print('neg_time:%f'%neg_time)

            

                # for c in range(K):
                #     #此处为句子中的每一个节点生成K个负例
                #     negative_node = np.random.choice(self.neg_nodeset, p=node_negative_distribution)
                #     #u_i=[node0,node0]
                #     #u_j=[neg_node0,neg_node1]
                #     u_i.append(node)
                #     u_j.append(negative_node)
                #     label.append(-1.)
                #从种群中选择负样本节点

                #sample_neg_deepwalk=list(np.random.choice(list(self.node_index.values()),size=k_2,p=self.node_degree))

                #sample_neg=list(np.random.choice(self.neg_nodeset[node],k_2,replace=False))
                for n_node in sample_neg:
                    u_i.append(node)
                    u_j.append(n_node)
                    label.append(-1.)

        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        #生成{node:embedding}字典
        return {node: embedding[self.node_index[node]] for node in self.nodes_raw}


    def get_embedding(self,embedding_node,u):
        #返回
        return embedding_node[u]





if __name__ == '__main__':
    graph_file = './data/wiki/Wiki_edgelist.txt'
    data_loader = DBLPDataLoader(graph_file=graph_file)
    a = np.random.rand(data_loader.num_of_nodes, 100)
    u_i, u_j, label = data_loader.fetch_batch(a,)
    print(len(u_i))
    print('\n---------------\n')
    print(len(u_j))
    print('\n---------------\n')
    print(len(label))





