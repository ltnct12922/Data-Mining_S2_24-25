import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load
        DataSource source = new DataSource("data/computed_insight_success_of_active_sellers.csv");
        Instances data = source.getDataSet();

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Clean missing values
        Instances noMissing = HandlemissValues.handel(data);
        // remove attribute
        Instances reduced = removeAttr.remove(noMissing, "2-3,6-11,13");

        // Normalize
        Instances normalized = NormalizeData.Normalize(reduced);

        Instances selected = AttrSelection.select(normalized);
        // Save result
        SaveData.save(selected, "output/clean_data.arff");
        // Split data
        SplitData.splitData(selected);
        System.out.println("Data preprocessing complete!");
        System.out.println("----Data Analysis----");
        DataSource output = new DataSource("output/clean_data.arff");
        Instances outputData = output.getDataSet();
        AttInst.dataAnalysis(outputData);
    }
}
