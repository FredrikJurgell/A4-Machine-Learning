import java.io.IOException;

/**
 * The Main class is the entry point of the program. It reads and analyzes two datasets:
 * the Banknote Authentication dataset and the Iris dataset.
 */
public class Main {
  /**
   * The main method is the entry point of the program.
   * It reads two datasets, the Banknote Authentication dataset and the Iris dataset,
   * and analyzes them using the analyzeDataset method.
   * 
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    try {
      // Path to the datasets
      String banknotePath = "banknote_authentication.csv";
      String irisPath = "Iris/iris.csv";

      // Reading the Banknote Authentication dataset
      System.out.println("Analyzing Banknote Authentication Dataset:");
      analyzeDataset(banknotePath, false);

      // Reading the Iris dataset
      System.out.println("\nAnalyzing Iris Dataset:");
      analyzeDataset(irisPath, true);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Analyzes a dataset given its file path and whether it has string labels.
   * It fits a NaiveBayes model to the dataset, makes predictions, and calculates accuracy.
   *
   * @param filePath       the file path of the dataset
   * @param hasStringLabels true if the dataset has string labels, false otherwise
   * @throws IOException if an I/O error occurs while reading the dataset
   */
  private static void analyzeDataset(String filePath, boolean hasStringLabels) throws IOException {
    float[][] X = CSVReader.readFeatures(filePath, hasStringLabels);
    int[] y = CSVReader.readLabels(filePath, hasStringLabels);

    NaiveBayes nb = new NaiveBayes();
    nb.fit(X, y);

    int[] predictions = nb.predict(X);
    float accuracy = NaiveBayes.accuracy_score(predictions, y);

    System.out.println("Accuracy: " + accuracy);
  }
}
