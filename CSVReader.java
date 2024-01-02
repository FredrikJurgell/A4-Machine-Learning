import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * The CSVReader class provides methods for reading features and labels from a CSV file.
 */
public class CSVReader {

  private static Map<String, Integer> labelMap = new HashMap<>();

  /**
   * Reads the features from a CSV file and returns them as a 2D float array.
   *
   * @param filePath        the path to the CSV file
   * @param hasStringLabels indicates whether the CSV file has string labels in the first column
   * @return a 2D float array containing the features from the CSV file
   * @throws IOException if an I/O error occurs while reading the file
   */
  public static float[][] readFeatures(String filePath, boolean hasStringLabels) throws IOException {
    List<float[]> records = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
      String line;
      boolean firstLine = true;

      while ((line = br.readLine()) != null) {
        if (firstLine) {
          firstLine = false;
          continue;
        }

        String[] values = line.split(",");
        int featureLength = hasStringLabels ? values.length - 1 : values.length - 1;
        float[] features = new float[featureLength];

        for (int i = 0; i < featureLength; i++) {
          features[i] = Float.parseFloat(values[i]);
        }
        records.add(features);
      }
    }
    return records.toArray(new float[0][]);
  }

  /**
   * Reads the labels from a CSV file.
   *
   * @param filePath        the path to the CSV file
   * @param hasStringLabels indicates whether the labels are string values
   * @return an array of labels as integers
   * @throws IOException if an I/O error occurs while reading the file
   */
  public static int[] readLabels(String filePath, boolean hasStringLabels) throws IOException {
    List<Integer> labels = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
      String line;
      boolean firstLine = true;
      int labelIndex = 0;

      while ((line = br.readLine()) != null) {
        if (firstLine) {
          firstLine = false;
          continue;
        }

        String[] values = line.split(",");
        String labelString = values[values.length - 1];

        if (hasStringLabels) {
          labelMap.putIfAbsent(labelString, labelIndex);
          labels.add(labelMap.get(labelString));
          labelIndex++;
        } else {
          labels.add(Integer.parseInt(labelString));
        }
      }
    }
    return labels.stream().mapToInt(i -> i).toArray();
  }
}
