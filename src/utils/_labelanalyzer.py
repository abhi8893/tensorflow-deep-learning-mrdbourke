import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LabelAnalyzer:

    """Analyze Labels for classification problem.

        Args:
            train_labels ([type]): train labels.
            test_labels ([type], optional): test labels. Defaults to None.
            classes ([type], optional): Actual class names or a mapping from class_labels to class_names. 
                                        Defaults to None.
    """
    
    def __init__(self, train_labels, test_labels=None, classes=None):

        train_dict = self._get_count_dict(train_labels)
        if test_labels is not None:
            test_dict = self._get_count_dict(test_labels)
        else:
            test_dict = {}

        self.__from_countdict_process(train_dict, test_dict, classes)
        self._make_count_df()

    @classmethod
    def from_countdict(cls, train_dict, test_dict, classes=None):
        obj = cls.__new__(cls)
        obj.__from_countdict_process(train_dict, test_dict, classes)
        return obj


    def __from_countdict_process(self, train_dict, test_dict=None, classes=None):

        self.train = train_dict
        self.__has_test_data = bool(test_dict)

        class_labels = self.train['label']

        if self.__has_test_data:
            self.test = test_dict
            class_labels = tuple(set(class_labels + self.test['label']))
        else:
            self.test = {}
        
        if classes is not None:
            if isinstance(classes, (tuple, list, np.ndarray)):
                assert len(classes) == len(class_labels)
                class_names = tuple(classes)

            elif isinstance(classes, dict):
                assert set(classes.keys()).intersection(class_labels) == set(class_labels)

                class_labels = tuple(classes.keys())
                class_names = tuple(classes.values())

        else:
            class_names = class_labels


        self.class_names = class_names
        self.class_labels = class_labels
        self.n_classes = len(self.class_labels)
        self.lab2nm = dict(zip(self.class_labels, self.class_names))
        self.nm2lab = {nm: lab for lab, nm in self.lab2nm.items()}
    

        self._make_count_df()



    
    @staticmethod
    def _get_count_dict(labels):
        unique, counts = np.unique(labels, return_counts=True)
        unique, counts = tuple(unique), tuple(counts)
        return dict(zip(['label', 'count'], [unique, counts]))


        
    def count(self, class_name, subset='train'):
        d = self.__getattribute__(subset)

        idx = d['label'].index(self.nm2lab[class_name])
        return d['count'][idx]
    
    def _make_count_df(self):
        traindf = pd.DataFrame(self.train)

        if not self.__has_test_data:
            countdf = traindf
        else:
            testdf = pd.DataFrame(self.test)
            countdf = pd.merge(traindf, testdf, how='outer', on='label', 
                            suffixes=('_train', '_test')).fillna(0).sort_values('label')
            
            countdf[['count_train', 'count_test']] = countdf[['count_train', 'count_test']].astype('Int64')
        
            countdf = countdf


        countdf = countdf.set_index('label').reindex(self.class_labels).reset_index().fillna(0)
        countdf['name'] = countdf['label'].map(self.lab2nm)

        first_cols = ['label', 'name']
        other_cols = list(countdf.columns[~countdf.columns.isin(first_cols)])
        
        self.countdf = countdf[first_cols + other_cols]

        
        
    def plot(self):

        ax = self.countdf.drop('label', axis=1).plot(x='name', kind='bar', stacked=True, figsize=(12, 4))

        if self.__has_test_data:
            title = 'Train vs Test label distribution'
            legend_labels = ['train', 'test']
        else:
            title = 'Train label distribution'
            legend_labels = ['train']

        ax.set_title(title, fontdict=dict(weight='bold', size=15))
        ax.legend(labels=legend_labels, bbox_to_anchor=(1.01, 0.6))
        plt.xticks(rotation=45)
        return ax