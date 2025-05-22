import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class HandlemissValues {
    public static Instances handel(Instances data) throws Exception {
        ReplaceMissingValues filter = new ReplaceMissingValues();
        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();
        int[] missingCounts = new int[numAttributes];
        for (int i = 0; i < numInstances; i++) {
            Instance instance = data.instance(i);
            for (int j = 0; j < numAttributes; j++) {
                if (instance.isMissing(j)) {
                    missingCounts[j]++;
                }
            }
        }
        System.out.println("missing Percentage (%):");
        for (int j = 0; j < numAttributes; j++) {
            double percentage = (missingCounts[j] * 100.0) / numInstances;
            if (percentage > 0) {
                System.out.printf("- %s: %d missing (%.2f%%)%n", data.attribute(j).name(), missingCounts[j],
                        percentage);
                System.out.println();
            }
        }
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }
}
