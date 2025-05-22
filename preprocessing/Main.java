import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public static void main(String[] args) throws Exception {
        // Load
        DataSource source = new DataSource("data/computed_insight_success_of_active_sellers.csv");
        Instances data = source.getDataSet();

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        // Analy data before processing data
        System.out.println("----Data Analysis raw data----");
        AttInst.dataAnalysis(data);
        // Clean missing values
        Instances noMissing = HandlemissValues.handel(data);
        // Removing Duplicate
        Instances removeDup = removeDuplicates.removeDup(noMissing);
        // remove attribute
        Instances reduced = removeAttr.remove(removeDup, "2-3,6-11,13");

        // Normalize
        Instances normalized = NormalizeData.Normalize(reduced);

        Instances selected = AttrSelection.select(normalized);
        // Save result
        SaveData.save(selected, "output/clean_data.arff");
        // Split data
        SplitData.splitData(selected);
        System.out.println("Data preprocessing complete!");
        // Analy after processing data
        System.out.println("----Data Analysis after processing data----");
        DataSource output = new DataSource("output/clean_data.arff");
        Instances outputData = output.getDataSet();
        AttInst.dataAnalysis(outputData);
    }
}
