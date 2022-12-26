from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class TextMining:
    def __init__(self, ngram: int = 1):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram), norm = 'l2')
        self.count_vectorizer = CountVectorizer()
    
    
    def vectorized_text(self, text_to_vectorize: list, count: bool = False):
        if count: res = self.count_vectorizer.fit_transform(text_to_vectorize)
        else: res = self.tfidf_vectorizer.fit_transform(text_to_vectorize)
        print(f'Shape of Sparse Matrix {res.shape}, type: {type(res)}')
        return res
    
    #@classmethod
    def get_features_names(self):
        return self.tfidf_vectorizer.get_feature_names_out()

    def lda_topic_modelling(self, encoded, topics:int) -> LatentDirichletAllocation():
        lda = LatentDirichletAllocation(n_components= topics, max_iter = 15, learning_method= "online",
                                        learning_offset= 5.,random_state= 1)
        lda.fit(encoded)
        return lda

    
    def word_cloud_dict(self, model: LatentDirichletAllocation) -> dict:
        features_names, res = self.tfidf_vectorizer.get_feature_names_out(), dict()
        for idx, topic in enumerate(model.components_):
            res[idx] = {}
            top_features_idx = topic.argsort()[::-1]
            top_features = [features_names[i] for i in top_features_idx]
            weights = topic[top_features_idx]
            for iel in range(len(weights)):
                res[idx][top_features[iel]] = int(weights[iel])
        return res

    @staticmethod
    def sort_coo(adj):
        coo_matrix=adj.tocoo()
        tuples=zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        sorted_items = sorted_items[:topn]
        score_vals, feature_vals, res = [], [], {}
        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
        for idx in range(len(feature_vals)):
            res[feature_vals[idx]]=score_vals[idx]
        print(f"Keywords found: {len(res)}")
        return res

'''
    ############################# LSA
    @staticmethod
    def latent_semantic_analysis(encoded, components: int) -> Tuple[TruncatedSVD,np.array]:
        svd = TruncatedSVD(n_components=components, n_iter=10)
        normalizer = Normalizer(norm='l2', copy=False)
        lsa = make_pipeline(svd, normalizer)
        svd_result = lsa.fit_transform(encoded)
        print(f"Explained variance of the SVD step: {svd.explained_variance_ratio_.sum()}%")
        return svd, svd_result

    @staticmethod
    def plot_lsa(svd_result: np.array, kmeans_model: KMeans, n_components: int, save: bool = False) -> None:
        svd_df=pd.DataFrame(svd_result, columns=[f'Component {i + 1}' for i in range(n_components)])
        svd_df['cluster']=kmeans_model.labels_
        fig, axes=plt.subplots(1, 3, figsize=(23, 10))
        axes[0].set_title(f'Latent Semantic Analysis 1-2\n', fontsize=25)
        axes[1].set_title('Latent Semantic Analysis 1-3\n', fontsize=25)
        axes[2].set_title('Latent Semantic Analysis 2-3\n', fontsize=25)
        sns.scatterplot(ax=axes[0], data=svd_df, x='Component 1', y='Component 2', hue='cluster', palette='tab10')
        sns.scatterplot(ax=axes[1], data=svd_df, x='Component 1', y='Component 3', hue='cluster', palette='tab10')
        sns.scatterplot(ax=axes[2], data=svd_df, x='Component 2', y='Component 3', hue='cluster', palette='tab10')
        fig.tight_layout()
        if save:
            plt.savefig(f'pics/lsa.png', format='png')
        plt.show()
        return results

    def get_wordcloud_lsa(self, svd_model, topics) -> list:
        svd_df = pd.DataFrame(svd_model.components_, columns=(self.tfidf_vectorizer.get_feature_names_out())).T
        res = []
        for i in range(topics):
            d = {}
            tmp = svd_df.loc[(svd_df[i] > 0.001), [i]][0:250].reset_index().sort_values(by=i, ascending=False)
            tmp[i] = tmp[i].apply(lambda x: x * 100)
            for a, x in tmp.values: d[a]=x
            res.append(d)
        return res
    
    def plot_lsa_topic(self, svd_model, topics, top_word, save: bool = False):
        fig, axes=plt.subplots(topics, 1, figsize=(20, 20))
        svd_df=pd.DataFrame(svd_model.components_, columns=(self.tfidf_vectorizer.get_feature_names_out())).T
        for i in range(topics):
            tmp = svd_df.loc[(svd_df[i] > 0.1), [i]][0:top_word].reset_index().sort_values(by=i, ascending=False)
            sns.barplot(data=tmp, y='index', x=i, palette=self.palette, ax=axes[i])
            axes[i].set_title(f'Topic {i} characterizing words', fontsize=20)
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
        fig.tight_layout()
        if save:
            plt.savefig(f'pics/lsa-topic.png', format='png',dpi=300)
        plt.show()
    
    ############################# K-MEANS CLUSTERING
    def kmeans_cluster(self, red_data:np.array, num_clusters: int = 3) -> KMeans:
        cluster_model = KMeans(n_clusters=num_clusters, init = "k-means++", max_iter = 100, n_init = 15)
        cluster_model.fit(red_data)
        return cluster_model

'''