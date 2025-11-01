"""
Multi-modal classifier for combining text and image features.
"""

from scipy.sparse import hstack, csr_matrix


class MultiModalClassifier:
    """
    Multi-modal classifier that combines text (TF-IDF) and image features.
    Compatible with sklearn interface for MLflow logging.
    """
    def __init__(self, vectorizer, image_features_train, classifier):
        self.vectorizer = vectorizer
        self.image_features_train = image_features_train
        self.classifier = classifier
        self.classes_ = None
    
    def fit(self, X_text, y):
        """
        Fit the model.
        X_text: array-like of text data (already aligned with image features)
        """
        # Vectorize text
        X_text_vec = self.vectorizer.fit_transform(X_text)
        
        # Combine text + image features
        X_combined = hstack([X_text_vec, csr_matrix(self.image_features_train)])
        
        # Train classifier
        self.classifier.fit(X_combined, y)
        self.classes_ = self.classifier.classes_
        return self
    
    def predict(self, X_text, image_features_test=None):
        """
        Predict on new data.
        X_text: array-like of text data
        image_features_test: numpy array of image features (must be provided)
        """
        if image_features_test is None:
            raise ValueError("image_features_test must be provided for prediction")
        
        # Vectorize text
        X_text_vec = self.vectorizer.transform(X_text)
        
        # Combine text + image features
        X_combined = hstack([X_text_vec, csr_matrix(image_features_test)])
        
        return self.classifier.predict(X_combined)
    
    def predict_proba(self, X_text, image_features_test=None):
        """Predict probabilities"""
        if image_features_test is None:
            raise ValueError("image_features_test must be provided for prediction")
        
        X_text_vec = self.vectorizer.transform(X_text)
        X_combined = hstack([X_text_vec, csr_matrix(image_features_test)])
        
        return self.classifier.predict_proba(X_combined)

