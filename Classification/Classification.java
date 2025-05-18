import weka.classifiers.trees.J48;
import weka.core.SerializationHelper;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

// 
public class Classification {
    public static String path = "clean_data.arff";
    private static DataSource source;
    private static Instances data;
    private static Evaluation eval;

    public static void main(String[] args) throws Exception {
        j48Classifier();
        naiveBayesClassifier();
        // randForestClassifier();
    }

    private static void j48Classifier() throws Exception {
        // 1. Load dataset
        source = new DataSource(path);
        data = source.getDataSet();

        // 2. Set class index (last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        // 3. Convert ALL numeric attributes (2…last) to nominal
        NumericToNominal num2nom = new NumericToNominal();
        num2nom.setAttributeIndices("2-last"); // convert listedproducts…urgencytextrate
                                               // :contentReference[oaicite:0]{index=0}
        num2nom.setInputFormat(data);
        Instances newData = Filter.useFilter(data, num2nom); // apply filter :contentReference[oaicite:1]{index=1}

        // 4. Build & configure J48 (C4.5) decision tree
        J48 tree = new J48();
        tree.setOptions(new String[] { "-C", "0.25", "-M", "2" }); // confidence=0.25, minNumObj=2
                                                                   // :contentReference[oaicite:2]{index=2}

        // 5. Evaluate with 10-fold cross-validation on filtered data
        eval = new Evaluation(newData);
        long start = System.currentTimeMillis();
        eval.crossValidateModel(tree, newData, 10, new Random(1));
        long end = System.currentTimeMillis();

        // 6. Train on full filtered data & save model
        tree.buildClassifier(newData);
        SerializationHelper.write("DecisionTree.model", tree);

        // 7. Output model & metrics
        System.out.println("=== J48 Decision Tree Model ===\n");
        System.out.println(tree);

        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        System.out.printf("Accuracy (pctCorrect): %.2f%%%n", eval.pctCorrect());
        System.out.printf("Weighted Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Weighted Recall: %.4f%n", eval.weightedRecall());
        System.out.printf("Runtime: %.3f sec%n", (end - start) / 1000.0);
        System.out.println("Model saved to DecisionTree.model");
    }

    private static void naiveBayesClassifier() throws Exception {
        // 1. Load dataset
        source = new DataSource(path);
        data = source.getDataSet();

        // 2. Set class index (last attribute)
        data.setClassIndex(data.numAttributes() - 1);

        // 3. Convert ALL numeric attributes (2…last) to nominal
        NumericToNominal num2nom = new NumericToNominal();
        num2nom.setAttributeIndices("2-last"); // convert from 2nd through last attrs
                                               // :contentReference[oaicite:0]{index=0}
        num2nom.setInputFormat(data);
        Instances newData = Filter.useFilter(data, num2nom); // apply filter :contentReference[oaicite:1]{index=1}

        // 4. Build & configure NaiveBayes
        NaiveBayes model = new NaiveBayes();
        model.setOptions(new String[] { "-K" }); // kernel estimator option :contentReference[oaicite:2]{index=2}

        // 5. Evaluate with 10-fold cross-validation on filtered data
        eval = new Evaluation(newData);
        long start = System.currentTimeMillis();
        eval.crossValidateModel(model, newData, 10, new Random(1));
        long end = System.currentTimeMillis();

        // 6. Train on full filtered data
        model.buildClassifier(newData);

        // 7. Output model & metrics
        System.out.println("=== Naive Bayes Model ===\n");
        System.out.println(model);

        System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        System.out.printf("Accuracy (pctCorrect): %.2f%%%n", eval.pctCorrect());
        System.out.printf("Weighted Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Weighted Recall: %.4f%n", eval.weightedRecall());
        System.out.printf("Runtime: %.3f sec%n", (end - start) / 1000.0);
    }

    private static void randForestClassifier() throws Exception {
        // 1. Load dataset
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();

        // 2. Remove merchantid (1st attribute)
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("1");
        removeFilter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeFilter);

        // 3. Convert class attribute (totalurgencycount) to nominal
        filteredData.setClassIndex(filteredData.numAttributes() - 1);
        NumericToNominal convertFilter = new NumericToNominal();
        convertFilter.setAttributeIndices("" + (filteredData.classIndex() + 1));
        convertFilter.setInputFormat(filteredData);
        Instances nominalData = Filter.useFilter(filteredData, convertFilter);

        // 4. Configure optimized Random Forest
        RandomForest rf = new RandomForest();
        rf.setOptions(new String[] {
                "-I", "100", // Number of trees
                "-depth", "5", // Max tree depth
                "-K", "2", // Features to consider at splits
                "-num-slots", "4" // Parallel execution
        });

        // 5. Evaluate with 10-fold cross-validation
        Evaluation eval = new Evaluation(nominalData);
        long startTime = System.currentTimeMillis();
        eval.crossValidateModel(rf, nominalData, 10, new Random(1));
        long endTime = System.currentTimeMillis();

        // 6. Train final model
        rf.buildClassifier(nominalData);

        // 7. Print results
        System.out.println("=== Optimized Random Forest ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        System.out.printf("Runtime: %.3f seconds\n", (endTime - startTime) / 1000.0);
    }
}
