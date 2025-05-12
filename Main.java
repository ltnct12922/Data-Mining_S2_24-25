import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        String path = "data/computed_insight_success_of_active_sellers.csv";
        File file = new File(path);

        if (!file.exists()) {
            System.out.println("File not exists: " + file.getAbsolutePath());
            return;
        }

        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();

        System.out.println("Loaded data: " + data.numInstances() + " instances.");
    }
}
