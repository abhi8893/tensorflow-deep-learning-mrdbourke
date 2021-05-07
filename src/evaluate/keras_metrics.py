from tensorflow.keras import backend as K


# Reimplement: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras 
class KerasMetrics:

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_val = true_positives / (possible_positives + K.epsilon())
        return recall_val


    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_val = true_positives / (predicted_positives + K.epsilon())
        return precision_val


    @classmethod
    def f1(cls, y_true, y_pred):

        precision = cls.precision(y_true, y_pred)
        recall = cls.recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))