
//import required classes
import weka.experiment.Stats;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AttInst {
    public static void dataAnalysis(Instances data) throws Exception {
        // get number of attributes (notice class is not counted)
        int numAttr = data.numAttributes();
        for (int i = 0; i < numAttr; i++) {
            // check if current attr is of type nominal
            if (data.attribute(i).isNominal()) {
                System.out.println("The " + i + "th Attribute is Nominal");
                System.out.println("The " + i + "th Attribute Name is " + data.attribute(i).name());
                // get number of values
                int n = data.attribute(i).numValues();
                System.out.println("The " + i + "th Attribute has: " + n + " values");
                System.out.println();
            }
            // get an AttributeStats object
            AttributeStats as = data.attributeStats(i);
            int dc = as.distinctCount;
            System.out.println("The " + i + "th Attribute has: " + dc + " distinct values");

            // get a Stats object from the AttributeStats
            if (data.attribute(i).isNumeric()) {
                System.out.println("The " + i + "th Attribute is Numeric");
                System.out.println("The " + i + "th Attribute Name is " + data.attribute(i).name());
                Stats s = as.numericStats;
                System.out.println("The " + i + "th Attribute has min value: " + s.min + " and max value: " + s.max);
                System.out.println("The mean: " + s.mean + " StdDev: " + s.stdDev);
                System.out.println();
            }
        }

    }
}