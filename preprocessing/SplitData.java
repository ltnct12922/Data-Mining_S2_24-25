import weka.core.Instances;
import weka.core.converters.ArffSaver;


import java.io.File;
import java.util.Random;

public class SplitData {
    public static void splitData(Instances data) throws Exception {

        // Randomize the dataset with a seed
        data.randomize(new Random(42));

        // Calculate indices for splitting
        int totalInstances = data.numInstances();
        int testSize = (int) Math.round(totalInstances * 0.2);
        int trainSize = totalInstances - testSize;
        int validationSize = (int) Math.round(trainSize * 0.2);
        trainSize -= validationSize;

        // Create empty sets for train, test, and validation
        Instances testData = new Instances(data, 0, testSize);
        Instances validationData = new Instances(data, testSize, validationSize);
        Instances trainData = new Instances(data, testSize + validationSize, trainSize);

        // Save the datasets to ARFF files
        saveInstances(trainData, "output/training_data.arff");
        saveInstances(testData, "output/test_data.arff");
        saveInstances(validationData, "output/validation_data.arff");
    }

    private static void saveInstances(Instances data, String fileName) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();
    }
}