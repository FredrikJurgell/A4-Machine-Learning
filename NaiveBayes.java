import java.util.*;

/**
 * The NaiveBayes class implements the Naive Bayes algorithm for classification.
 * It is used to train a model on a given dataset and make predictions on new data.
 */
public class NaiveBayes {
  private Map<Integer, Integer> classCounts;
  private Map<Integer, Float> classProbabilities;
  private Map<Integer, List<Float>> featureSums;
  private Map<Integer, List<Float>> featureSquaredSums;
  private int numFeatures;

  /**
   * This class represents a Naive Bayes classifier.
   */
  public NaiveBayes() {
    classCounts = new HashMap<>();
    classProbabilities = new HashMap<>();
    featureSums = new HashMap<>();
    featureSquaredSums = new HashMap<>();
  }

  /**
   * Fits the Naive Bayes classifier to the given training data.
   * 
   * @param X The feature matrix of shape [numSamples, numFeatures].
   * @param y The target labels of shape [numSamples].
   */
  public void fit(float[][] X, int[] y) {
    numFeatures = X[0].length;
    int numSamples = X.length;

    // Initialize data structures
    for (int i = 0; i < numSamples; i++) {
      int label = y[i];
      classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);

      if (!featureSums.containsKey(label)) {
        featureSums.put(label, new ArrayList<>(Collections.nCopies(numFeatures, 0.0f)));
        featureSquaredSums.put(label, new ArrayList<>(Collections.nCopies(numFeatures, 0.0f)));
      }

      for (int j = 0; j < numFeatures; j++) {
        float featureValue = X[i][j];
        featureSums.get(label).set(j, featureSums.get(label).get(j) + featureValue);
        featureSquaredSums.get(label).set(j, featureSquaredSums.get(label).get(j) + featureValue * featureValue);
      }
    }

    for (Integer label : classCounts.keySet()) {
      classProbabilities.put(label, (float) classCounts.get(label) / numSamples);
    }
  }

  /**
   * Predicts the class labels for the given feature vectors.
   *
   * @param X The feature vectors for which to predict the class labels.
   * @return An array of predicted class labels.
   */
  public int[] predict(float[][] X) {
    int[] predictions = new int[X.length];

    for (int i = 0; i < X.length; i++) {
      float[] features = X[i];
      int predictedClass = -1;
      double maxProb = Double.NEGATIVE_INFINITY;

      for (Integer label : classCounts.keySet()) {
        double prob = Math.log(classProbabilities.get(label));

        for (int j = 0; j < numFeatures; j++) {
          float mean = featureSums.get(label).get(j) / classCounts.get(label);
          float variance = featureSquaredSums.get(label).get(j) / classCounts.get(label) - mean * mean;
          prob += Math.log(gaussianProbability(features[j], mean, variance));
        }

        if (prob > maxProb) {
          maxProb = prob;
          predictedClass = label;
        }
      }

      predictions[i] = predictedClass;
    }

    return predictions;
  }

  /**
   * Calculates the probability of a given value using the Gaussian distribution.
   *
   * @param x the value for which the probability is calculated
   * @param mean the mean of the distribution
   * @param variance the variance of the distribution
   * @return the probability of the given value
   */
  private double gaussianProbability(float x, float mean, float variance) {
    double exponent = Math.exp(-((x - mean) * (x - mean)) / (2 * variance));
    return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
  }

  /**
   * Calculates the accuracy score of a set of predictions compared to the true labels.
   * The accuracy score is defined as the number of correct predictions divided by the total number of predictions.
   *
   * @param preds the array of predicted labels
   * @param y the array of true labels
   * @return the accuracy score as a float value between 0 and 1
   */
  public static float accuracy_score(int[] preds, int[] y) {
    int correct = 0;
    for (int i = 0; i < preds.length; i++) {
      if (preds[i] == y[i]) {
        correct++;
      }
    }

    return (float) correct / preds.length;
  }
}
