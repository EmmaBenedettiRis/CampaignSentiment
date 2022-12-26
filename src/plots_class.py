from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from nltk.probability import FreqDist
from src.params import party_leaders
from src.text_mining_class import TextMining
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import operator



class PlotsClass():

    def __init__(self, lab_threshold: int = None, col_threshold: int=None):
        self.label_threshold=lab_threshold
        self.color_threshold = col_threshold
        self.leaders = party_leaders
        self.height = 7
        self.width = 2*self.height
        self.colors_palette = 'tab10' #'Blues_r'
        #self.colors_palette = list(mcolors.TABLEAU_COLORS.values())
        self.title_ft = 16
         
        
    def plot_tweet_chrono(self, df: pd.DataFrame, title_pre: str = "Distribution of Tweets during the Election Campaign\n", title_post: str = "Distribution of Tweets after the Election Campaign\n",full:bool=True,save:bool=True):
        """
        Plots chronological distribution of tweets for the two months before and after the election date.
        """
        ## initialize parameters
        h = self.height     # height
        w = self.width      # width
        sns.set_theme(context= 'paper',
                      style='white',
                      #palette=sns.color_palette("hls", len(self.leaders))
                      palette= self.colors_palette
                    )

        #fig = plt.figure(figsize=(w,h),dpi=300)
        fig, axes=plt.subplots(1, 1, figsize=(w, h))

        tmp= self.subselect_and_sort_df(df,["Date", "Leader Name"],sort_by=["Date", "Leader Name"], full=full)
        daterange,idx = self.get_daterange_and_idx(tmp)
        
        #tmps = [tmp1, tmp2]
        #idxs = [idx1, idx2]
        #dateranges = [daterange1, daterange2]
        #titles = [title_pre, title_post]

        sns.histplot(data=tmp, x="Date", hue = "Leader Name", multiple="stack", palette=self.colors_palette)
        plt.title(title_pre, fontsize=self.title_ft)
        plt.xticks(ticks = range(0, len(daterange), idx))
        plt.xlabel("Date")
        plt.ylabel("Total Tweets")

        fig.tight_layout()
        if save:
            plt.savefig('pics/tweets_elections.png', format='png', dpi=300)
        plt.show()
    '''
    def plot_tweet_chrono(self, df: pd.DataFrame, title_pre: str = "Distribution of Tweets during the Election Campaign", title_post: str = "Distribution of Tweets after the Election Campaign",save:bool=True):
        """
        Plots chronological distribution of tweets for the two months before and after the election date.
        """
        ## initialize parameters
        h = self.height     # height
        w = self.width      # width
        sns.set_theme(context= 'paper',
                      style='white',
                      #palette=sns.color_palette("hls", len(self.leaders))
                      palette= self.colors_palette
                    )

        #fig = plt.figure(figsize=(w,h),dpi=300)
        fig, axes=plt.subplots(2, 1, figsize=(w, h), sharey = True)

        tmp1, tmp2 = self.subselect_and_sort_df(df,["Date", "Leader Name"],sort_by=["Date", "Leader Name"], full=False)
        daterange1,idx1 = self.get_daterange_and_idx(tmp1)
        daterange2,idx2 = self.get_daterange_and_idx(tmp2)
        
        tmps = [tmp1, tmp2]
        idxs = [idx1, idx2]
        dateranges = [daterange1, daterange2]
        titles = [title_pre, title_post]

        for i in range(len(tmps)):
            p = sns.histplot(data=tmps[i], x="Date", hue = "Leader Name", multiple="stack", ax=axes[i], palette=self.colors_palette)
            axes[i].set_title(titles[i], fontsize=self.title_ft)
            axes[i].set_xticks(ticks = range(0, len(dateranges[i]), idxs[i]))
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Total Tweets")

        #############################################################
        fig.tight_layout()
        if save:
            plt.savefig('pics/tweets_pre_post.png', format='png', dpi=300)
        plt.show()
    '''
    def plot_followers_count(self, df: pd.DataFrame, save:bool=False):
        '''
        Plot number of followers for each leader.
        '''
        h,w = 5, 5
        tmp = self.subselect_and_sort_df(df, ["Date", "Leader Name", "Followers"])
        daterange, idx = self.get_daterange_and_idx(tmp)
        fig = plt.figure(figsize=(w,h))
        
        sns.lineplot(data=tmp, x = "Date", y = "Followers", hue = "Leader Name", seed = 42)
        plt.title("Number of Followers by Leader\n", fontsize=self.title_ft)
        plt.xticks(ticks = range(0, len(daterange), idx))
        
        ## plot red line mark for election date (2022-09-25)
        datemark = np.where(daterange=='2022-09-25')[0][0]
        plt.vlines(x=datemark,ymin = 0.0, ymax= 3.5*(10**6), linestyles= 'dashed', color="crimson", label="Election Date")

        fig.tight_layout()
        if save:
            plt.savefig('pics/followers_count.png', format='png', dpi=300)
        plt.show()
    
    @staticmethod
    def subselect_and_sort_df(df:pd.DataFrame, cols: list(), sort_by:str or list="Date",full:bool=True, date_div = dt(2022, 9,25)) -> pd.DataFrame:
        df = df[cols]
        df = df.sort_values(by = sort_by)
        if full: return df
        else:
            df_pre = df[df["Date"] <= date_div]
            df_post = df[df["Date"] > date_div]
            return df_pre, df_post
    
    @staticmethod
    def get_daterange_and_idx(df=pd.DataFrame, idx_div = 5):
        df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')
        dr = np.unique(df["Date"])
        idx = len(dr)//idx_div
        return dr, idx
    
    def plot_lda_topic(self, model: LatentDirichletAllocation, topics: int,cls: TextMining(), n_top_words: int, save: bool = False) -> None:
        if topics < 20: h = 5*topics
        elif topics >= 20 and topics < 30: h = 1.75* topics
        else: h = 65
        fig, axes = plt.subplots(topics, 1, figsize=(self.width,h))
        features_names = cls.get_features_names()
        for idx, topic in enumerate(model.components_):
            top_features_idx = topic.argsort()[:-n_top_words -1: -1]
            top_features = [features_names[n] for n in top_features_idx]
            weights = topic[top_features_idx]
            sns.barplot(y=[feature.upper() for feature in top_features], x=weights,ax=axes[idx], palette=self.colors_palette)
            axes[idx].tick_params(axis='y', which='minor', labelsize=7)
            axes[idx].set_title(f'Most characterizing words for topic {idx+1}\n', fontsize=self.title_ft)
            axes[idx].set_xticks([])
            #sns.set_style('white')
        fig.tight_layout()
        if save:
            plt.savefig(f'pics/lda_topics.png', format='png', dpi=300)
        plt.show()
    
    def get_word_size(self, word:str, freq_dist:FreqDist) -> int:
        return freq_dist.N(word)
    
    def get_node_size(self, graph:nx.Graph) -> list:
        return [self.get_word_size(w) for w in graph.nodes()]

    def get_labels(self, graph:nx.Graph) -> dict:
        labels = dict()
        for node in graph.nodes():
            if self.get_word_size(node) > self.lab_threshold: labels[node.lower()] = node
        return labels

    def get_node_threshold_color(self,graph:nx.Graph):
        return [['#61a5c2', '#012a4a'][self.get_word_size(node) > self.color_threshold] for node in graph.nodes()]
    
    def get_node_col_cluster(self, labels_: list, random: bool=False, colors: list=False):
        res= []
        if random:
            tmp = dict()
            for label in np.unique(labels_):
                tmp[label] = self.colors_palette[np.random.randint(0, len(self.colors_palette))]
            for label in labels_:
                res.append(tmp[label])
        elif colors:
            for label in labels_:
                res.append(colors[label])
        else:
            for label in labels_:
                res.append(self.colors_palette[label])
        return res

    def get_node_size_centrality_labs(self, graph:nx.Graph, centr:dict, mult_fact: int,freq_dist:FreqDist=None, upper:bool=False, alt_nodesize:int=15):
        nodesize, labs = list(), dict()
        for node in graph.nodes():
            tmp = set()
            if node in centr:
                if freq_dist: nodesize.append(freq_dist[node]*mult_fact)
                else: nodesize.append(freq_dist[node]*mult_fact)
                if upper: labs[node.upper()] = node.upper()
                else: labs[node]=node
                tmp.add(node)
            else: nodesize.append(alt_nodesize)
        return nodesize, labs
    
    @staticmethod
    def extract_top_centrality_words(centrality, percentage):
        res = dict()
        for c in centrality:
            q, p = 0, (len(c)/100 * percentage)
            for item in c.items():
                if q < p:
                    if item[0] in res: res[item[0]] = item[1]
                    #    if res[item[0]] < item[1]: res[item[0]] = item[1]
                    #else: res[item[0]]=item[1]
                q+=1
        return res
    
    @staticmethod
    def create_graph_from_top_centrality(graph, res):
        tmp = []
        for (u,v,d) in graph.edges(data=True):
            if u in res or v in res:
                tmp.append((u,v, dict(count=d["count"])))
        return nx.Graph(tmp)
    
    @staticmethod
    def graph_filtered_dist(df:pd.DataFrame, distr: FreqDist, threshold:int, obj: str='tweet') -> nx.Graph:
        def check_threshold(w, distr:FreqDist, val:int):
            return distr.get(w) > val
        res = nx.Graph()
        if obj=='tweet': bag = df['Tokenized_Text']
        else: bag = df
        for text in bag:
            if (text):
                for w1 in text:
                    if check_threshold(w=w1, distr=distr, val=threshold):
                        for w2 in text:
                            if check_threshold(w=w2, distr=distr, val=threshold):
                                if w1 != w2:
                                    if not res.has_edge(w1,w2): res.add_edge(w1, w2, count = 1)
                                    else: res[w1][w2]["count"] +=1
        return res

    def plot_main_centrality(self, graph:nx.Graph, res, freq_dist:FreqDist=None, labels_:list=None, mult_factor: int = 5, save: bool = False, k=2, i=50, w='count', ka=True, c=None, a=0.1, s:int=111):
        fig, axes= plt.subplots(1,1, figsize=(self.width, self.width))
        plt.style.use('seaborn-white')
        if ka: layout = nx.kamada_kawai_layout(graph, weight=w)
        else: layout = nx.spring_layout(graph, weight=w, k=k, iterations=i, seed=s)
        node_sizes, labels = self.get_node_size_centrality_labs(graph, res, mult_factor, freq_dist)

        nx.draw_networkx_nodes( G=graph,
                                pos=layout,
                                cmap=plt.get_cmap(self.colors_palette),
                                node_size=node_sizes,
                                node_color='#03045e' if not labels_.any() else self.get_node_col_cluster(labels_, colors=c),
                                ax=axes,
                                alpha=0.8)

        nx.draw_networkx_edges( G=graph,
                                edge_color='black',
                                pos=layout,
                                width = [graph[u][v]['count'] / 7 for u,v in graph.edges],
                                alpha=a)

        nx.draw_networkx_labels(graph,
                                pos=layout,
                                labels=labels,
                                font_size=14,
                                ax=axes,
                                font_color='#000000',
                                verticalalignment='top')
        axes.set_axis_on()
        axes.grid(False)
        plt.title('Semantic Network Representation of most common Keywords\n', fontsize=self.title_ft)
        if save:
            plt.savefig('pics/main_centrality.png', format='png', dpi=300)
        plt.show()


    def plot_wordcloud(self, data, n_topics:int=2, save: bool = False):
        palette=self.colors_palette
        fig, axes=plt.subplots(1, n_topics, figsize=(self.height, self.width))
        wordcloud=WordCloud(margin=10,
                            background_color='white',
                            colormap=palette,
                            width=400, height=200,
                            max_words=150)
        for i in range(n_topics):
            word_cloud=wordcloud.generate_from_frequencies(data[i])
            axes[i].axis('off')
            axes[i].set_title(f'Characterizing words for topic {i+1}\n', fontsize=16)
            axes[i].imshow(word_cloud)
        ######################################################
        fig.tight_layout()
        if save:
            plt.savefig(f'pics/wordcloud.png', format='png')
        plt.show()
    
    @staticmethod
    def spectral_clustering(graph: nx.Graph, n_cluster:int=2, k: int=None, gamma: float=1.0, w='count',rs:int=111, check:bool=False):
        adj_mat = nx.to_numpy_array(graph, weight=w)
        if k: spectral_clust = SpectralClustering(n_cluster, affinity='precomputed_nearest_neighbors', n_init=100, assign_labels='discretize', gamma=gamma, random_state=rs)
        else: spectral_clust = SpectralClustering(n_cluster, affinity='rbf', assign_labels='discretize', gamma=gamma, random_state=rs)
        spectral_clust.fit(adj_mat)
        c = np.unique(spectral_clust.labels_, return_counts=True)[1].tolist()
        s= silhouette_score(adj_mat, spectral_clust.labels_)
        if check: return s
        else:
            print(f"Cluster distribution: {c}\tSilhouette score: {round(s,4)}")
            return spectral_clust, spectral_clust.labels_
    
    @staticmethod
    def keep_min_degree(graph: nx.Graph, min_deg:int=2) -> nx.Graph:
        res=nx.Graph.copy(graph)
        for c in graph.degree:
            if c[1] < min_deg: res.remove_node(c[0])
        return res

    @staticmethod
    def get_set_top_words(centrality, top:int=20):
        res = set()
        for i in centrality:
            tmp=dict(sorted(i.items(), key=operator.itemgetter(1), reverse=True)[:top])
            for i in tmp.keys(): res.add(i)
        return res

    @staticmethod
    def extend_top_word(graph:nx.Graph, threshold:int, res):
        ret = res.copy()
        for u,v in graph.edges:
            if graph[u][v]['count'] > threshold:
                ret.add(u)
                ret.add(v)
        return ret

    @staticmethod
    def keep_connected_components(graph: nx.Graph, min_deg:int):
        for c in list(nx.connected_components(graph)):
            if len(c) < min_deg:
                for n in c:
                    graph.remove_node(n)